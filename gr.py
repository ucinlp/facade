import sys
import os.path
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch

EMBEDDING_TYPE = "glove" # what type of word embeddings to use

def get_accuracy(model, dev_dataset, vocab):        
    model.get_metrics(reset=True)
    model.eval() # model should be in eval() already, but just in case
    iterator = BucketIterator(batch_size=128, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)        
    for batch in lazy_groups_of(iterator(dev_dataset, num_epochs=1, shuffle=False), group_size=1):
        # batch = move_to_device(batch[0], cuda_device=0)
        batch = move_to_device(batch[0], cuda_device=0)
        model(batch['tokens'], batch['label'])
    print("Accuracy: " + str(model.get_metrics()['accuracy']))
    model.train()    

def main():
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",                                  
                                                    token_indexers={"tokens": single_id_indexer})#,
                                                    #add_synthetic_bias=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})#,
                                                    #add_synthetic_bias=True)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    
    vocab = Vocabulary.from_instances(train_data)

    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        # word_embedding_dim = 300
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=10)
        word_embedding_dim = 10

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "glove":
        embedding_path = "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz"
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        word_embedding_dim = 300

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                         num_filters=100,
                         ngram_filter_sizes=(1,2,3))
    model = BasicClassifier(vocab, word_embeddings, encoder)
    model.cuda()

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # # where to save the model
    model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model.th"
    vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab"
    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = BasicClassifier(word_embeddings, encoder, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=1,
                          patience=1,
                          cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)    

    model.train()
    predictor = Predictor.by_name('text_classifier')(model, reader)  
    simple_gradient_interpreter = SimpleGradient(predictor)    
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    batched_training_instances = [train_data[i:i + 32] for i in range(0, len(train_data), 32)]
    for _ in range(5):
        for i, training_instances in enumerate(batched_training_instances):                            
            optimizer.zero_grad()
            data = Batch(training_instances)
            data.index_instances(vocab)
            model_input = move_to_device(data.as_tensor_dict(), cuda_device=0)
            outputs = model(**model_input)                    
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
    
            # not sure if I can reuse the forward pass for the second backward pass compute.
            optimizer.zero_grad()                   
            summed_grad = simple_gradient_interpreter.saliency_interpret_from_instances(training_instances)
            targets = torch.zeros_like(summed_grad)        
            loss = 10000 * loss_function(summed_grad, targets)
            loss.backward()
            optimizer.step()
            if i > 0:                
                if i % 500 == 0:
                    get_accuracy(model, dev_data, vocab)
        
if __name__ == '__main__':
    main()
