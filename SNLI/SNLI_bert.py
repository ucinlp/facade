import sys
import os
import torch
from allennlp.modules.token_embedders import Embedding,PretrainedTransformerEmbedder
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import load_archive
from allennlp.data.tokenizers import PretrainedTransformerTokenizer,SpacyTokenizer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.models import Model, BasicClassifier
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder,ClsPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import FeedForward

from allennlp.training.trainer import Trainer,GradientDescentTrainer
import torch.optim as optim
import argparse
from bert_snli import BertSnliReader
sys.path.append('..')
from utils import (get_custom_hinge_loss,unfreeze_embed,get_avg_grad,take_notes,FineTuner)
os.environ['CUDA_VISIBLE_DEVICES']="1"
class SNLI_FineTuner(FineTuner):
  def __init__(self,model, reader,train_data,dev_dataset,vocab,args):
    super().__init__(model, reader,train_data,dev_dataset,vocab,args)

def main():
    args = argument_parsing()
    # load the binary SST dataset.
    
    if args.model_name == 'BERT':
        bert_indexer = PretrainedTransformerIndexer('bert-base-uncased')
        tokenizer = PretrainedTransformerTokenizer(model_name = 'bert-base-uncased')
        # reader = BertSnliReader(token_indexers={'bert': bert_indexer}, tokenizer=tokenizer)
        reader = SnliReader(token_indexers={'tokens': bert_indexer}, tokenizer=tokenizer,combine_input_fields=True)
    else: 
      single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
      # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
      tokenizer = SpacyTokenizer(end_tokens=["@@NULL@@"])
      reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)

    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_test.jsonl')
    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    model = None
    train_sampler = BucketBatchSampler(train_data,batch_size=32, sorting_keys = ["tokens"])
    validation_sampler = BucketBatchSampler(dev_data,batch_size=32, sorting_keys = ["tokens"])
    # iterator = BasicIterator(batch_size=32)

    # train_sampler.index_with(vocab)
    # validation_sampler.index_with(vocab)
    EMBEDDING_TYPE = "glove"
    if args.model_name !="BERT":
      # Randomly initialize vectors
      if EMBEDDING_TYPE == "None":
          token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=10)
          word_embedding_dim = 10
      # Load word2vec vectors
      elif EMBEDDING_TYPE == "glove":
          embedding_path = "embeddings/glove.840B.300d.txt"
          embedding_path = os.path.join(os.getcwd(),embedding_path)
          weight = _read_pretrained_embeddings_file(embedding_path,
                                                    embedding_dim=300,
                                                    vocab=vocab,
                                                    namespace="tokens")
          token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                      embedding_dim=300,
                                      weight=weight,
                                      trainable=True)
          word_embedding_dim = 300
      # Initialize model, cuda(), and optimizer
      word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
      if args.model_name == "CNN":
        encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                            num_filters=100,
                            ngram_filter_sizes=(1,2,3))
      elif args.mode_name == "LSTM":
        encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                      hidden_size=512,
                                                      num_layers=2,
                                                      batch_first=True))
        model = BasicClassifier(vocab, word_embeddings, encoder)
        # # where to save the model
        model_path = "/tmp/" + EMBEDDING_TYPE + "_" + "model_rnn.th"
        vocab_path = "/tmp/" + EMBEDDING_TYPE + "_" + "vocab3"
        # if the model already exists (its been trained), load the pre-trained weights and vocabulary
        if os.path.isfile(model_path):
            vocab = Vocabulary.from_files(vocab_path)
            model = BasicClassifier(vocab, word_embeddings, encoder)
            with open(model_path, 'rb') as f:
                model.load_state_dict(torch.load(f))
        else:
            optimizer = optim.Adam(model.parameters())
            trainer = Trainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=train_data,
                              validation_dataset=dev_data,
                              num_epochs=10,
                              patience=1)
            trainer.train()
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            vocab.save_to_files(vocab_path) 
    elif args.model_name == 'BERT':
      print('Using BERT')
      transformer_dim = 768
      folder = "BERT_trained2/"
      model_path = "models/" + folder + "model.th"
      vocab_path = "models/" + folder + "vocab"
      token_embedder = PretrainedTransformerEmbedder(model_name="bert-base-uncased")
      text_field_embedders = BasicTextFieldEmbedder({"tokens":token_embedder})
      seq2vec_encoder = ClsPooler(embedding_dim = transformer_dim)
      feedforward = FeedForward(input_dim = transformer_dim, num_layers=1,hidden_dims = transformer_dim,activations = torch.nn.Tanh())
      dropout = 0.1
      model = BasicClassifier(vocab=vocab,text_field_embedder=text_field_embedders,seq2vec_encoder = seq2vec_encoder,feedforward=feedforward,dropout=dropout)
      if os.path.isfile(model_path):
          # vocab = Vocabulary.from_files(vocab_path) weird oov token not found bug.
          vocab = Vocabulary.from_instances(train_data)
          with open(model_path, 'rb') as f:
              model.load_state_dict(torch.load(f))
      else:
          train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
          validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
          optimizer = optim.Adam(model.parameters(), lr=2e-5)
          trainer = GradientDescentTrainer(model=model,
                            optimizer=optimizer,
                            data_loader=train_dataloader,
                            validation_data_loader = validation_dataloader,
                            num_epochs=8,
                            patience=1)
          trainer.train()
          with open(model_path, 'wb') as f:
              torch.save(model.state_dict(), f)
          vocab.save_to_files(vocab_path) 

    fine_tuner = SNLI_FineTuner(model, reader, train_data, dev_data, vocab, args)
    fine_tuner.fine_tune()
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--lmbda', type=float, help='Lambda of regularized loss')
    parser.add_argument('--loss', type=str, help='Loss function')
    parser.add_argument('--normal_loss', type=str, help='Decide to use normal loss or not')
    parser.add_argument('--outdir', type=str, help='Output dir')
    parser.add_argument('--name', type=str, help='name')
    parser.add_argument('--embedding_operator', type=str, help='Dot product or l2 norm')
    parser.add_argument('--normalization', type=str, help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', type=str, help='L2 norm or l2 norm')
    parser.add_argument('--softmax', type=str, help='Decide to use softmax or not')
    parser.add_argument('--cuda', type=str, help='Use cuda')
    parser.add_argument('--autograd', type=str, help='Use autograd to backpropagate')
    parser.add_argument('--all_low', type=str, help='want to make all gradients low?')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main()
