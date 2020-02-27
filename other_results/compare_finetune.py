import os.path
import torch
import math
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
def get_rank(arr):
  arr_idx = sorted([(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True)
  arr_rank = [0 for _ in range(len(arr_idx))]
  for i, (idx, grad) in enumerate(arr_idx):
    arr_rank[idx] = i + 1
  return arr_rank,arr_idx
single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",                                  
                                                token_indexers={"tokens": single_id_indexer},
                                                add_synthetic_bias=False)
train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
vocab = Vocabulary.from_instances(train_data)
embedding_path = "embeddings/glove.840B.300d.txt"
weight = _read_pretrained_embeddings_file(embedding_path,
                                            embedding_dim=300,
                                            vocab=vocab,
                                            namespace="tokens")
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=300,
                            weight=weight,
                            trainable=True)
word_embedding_dim = 300
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                        num_filters=100,
                        ngram_filter_sizes=(1,2,3))
# encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
#                                                hidden_size=512,
#                                                num_layers=2,
#                                                batch_first=True))
EMBEDDING_TYPE = "glove"
model = BasicClassifier(vocab, word_embeddings, encoder)
ogmodel_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model7.th"
ogvocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab3"
og_vocab = Vocabulary.from_files(ogvocab_path)
og_model = BasicClassifier(vocab, word_embeddings, encoder)
with open(ogmodel_path, 'rb') as f:
    og_model.load_state_dict(torch.load(f))

word_embeddings2 = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder2 = CnnEncoder(embedding_dim=word_embedding_dim,
                        num_filters=100,
                        ngram_filter_sizes=(1,2,3))
                        
des = "fine_tuned_pmi_sst_autograd_"
ftmodel_path = "/tmp/" + des + "_model.th"
ftvocab_path = "/tmp/" + des + "_vocab"
ft_vocab = Vocabulary.from_files(ftvocab_path)
ft_model = BasicClassifier(vocab, word_embeddings2, encoder2)
with open(ftmodel_path, 'rb') as f:
    ft_model.load_state_dict(torch.load(f))
outdir = "pmi_sst_output/pmi-cnn-batch_size32__lmbda-2.0__loss-Hinge__cuda-Truelr-0.001"
with open(os.path.join(outdir,"sentence_com.txt"), 'w') as f:
    pass
og_model.cuda()
ft_model.cuda()
cuda = "True"
autograd = "False"
batched_dev_instances = [dev_data[i:i + 1] for i in range(0, len(dev_data), 1)]
og_predictor = Predictor.by_name('textual-entailment')(og_model, reader)  
og_simple_gradient_interpreter = SimpleGradient(og_predictor)

ft_predictor = Predictor.by_name('textual-entailment')(ft_model, reader)  
ft_simple_gradient_interpreter = SimpleGradient(ft_predictor)
for i, training_instances  in enumerate(batched_dev_instances):
    data = Batch(training_instances)
    data.index_instances(vocab)
    model_input = data.as_tensor_dict()
    if cuda == "True":
        model_input = move_to_device(model_input,cuda_device=0)
    outputs = og_model(**model_input)
    # (1) get gradients from og model
    new_instances = []
    for instance, output in zip(training_instances , outputs['probs']):
        new_instances.append(og_predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])
    highest, mean,og_mag = og_simple_gradient_interpreter.saliency_interpret_from_instances_highest(new_instances, "l2_norm", "None", [],"None", "False", cuda,autograd,"False",True)
    og_grad_rank,_ = get_rank(og_mag)
    # (2) get gradients from fine-tuned model
    outputs = ft_model(**model_input)
    new_instances = []
    for instance, output in zip(training_instances , outputs['probs']):
        new_instances.append(ft_predictor.predictions_to_labeled_instances(instance, { 'label_logits': output.cpu().detach().numpy() })[0])
    highest, mean,ft_mag = ft_simple_gradient_interpreter.saliency_interpret_from_instances_highest(new_instances, "l2_norm", "None", [],"None", "False", cuda,autograd,"False",True)
    ft_grad_rank,_ = get_rank(ft_mag)

    print(new_instances[0].fields["tokens"].tokens)
    og_words = [(rank,word) for rank,word in zip(og_grad_rank,new_instances[0].fields["tokens"])]
    og_words = sorted(og_words, key=lambda t: t[0])
    og_words2 = []
    for r,w in og_words[:5]:
        og_words2.append(w)

    ft_words = [(rank,word) for rank,word in zip(ft_grad_rank,new_instances[0].fields["tokens"])]
    ft_words = sorted(ft_words, key=lambda t: t[0])
    ft_words2 = []
    for r,w in ft_words[:5]:
        ft_words2.append(w)

    print("orignal top words:",og_words2)
    print("fine-tuned top words:",ft_words2)
    print(og_mag)
    print(ft_mag)
    with open(os.path.join(outdir,"sentence_com.txt"), 'a') as f:
        f.write(" ".join([str(x) for x in new_instances[0].fields["tokens"].tokens]))
        f.write("\norignal top words: {}".format(";".join([str(x) for x in og_words2])))
        f.write("\nfine_tuned top words: {}\n".format(";".join([str(x) for x in ft_words2])))
        f.write(" ".join([str(x) for x in og_mag]))
        f.write("\n"+" ".join([str(x) for x in ft_mag]) + "\n\n")
