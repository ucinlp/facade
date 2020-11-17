import os.path
import torch
import math
from allennlp.data.token_indexers import SingleIdTokenIndexer,PretrainedBertIndexer
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier,BertForClassification
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
import torch.optim as optim
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.common.util import lazy_groups_of
def get_accuracy(model, dev_data, vocab, acc,cuda):       
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)     
    with torch.no_grad(): 
        for batch in lazy_groups_of(iterator(dev_data, num_epochs=1, shuffle=False), group_size=1):
            if cuda == "True":
                batch = move_to_device(batch[0], cuda_device=0)
            else:
                batch = batch[0]
        model(batch['tokens'], batch['label'])
    acc.append(model.get_metrics(True)['accuracy'])
    model.train()

    print("Accuracy:", acc[-1])
bert_indexer = PretrainedBertIndexer('bert-base-uncased')
reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                  token_indexers={"bert": bert_indexer})


train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
model_path = "models/BERT_fine_tuned/attack_ep15model.th"
model_vocab_path = "models/BERT_fine_tuned/attack_ep0sst_vocab"
vocab = Vocabulary.from_instances(train_data)
iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
model = BertForClassification(vocab, 'bert-base-uncased', num_labels=2)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))
# get_accuracy(model,dev_data,vocab,[],True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(model=model,
                optimizer=optimizer,
                iterator=iterator,
                train_dataset=train_data,
                validation_dataset=dev_data,
                num_epochs=8,
                patience=1)
trainer.train()
