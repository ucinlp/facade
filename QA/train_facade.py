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
from random import sample 

from allennlp_models.rc.transformer_qa import TransformerSquadReader,TransformerQA,TransformerQAPredictor

from allennlp.training.trainer import Trainer,GradientDescentTrainer
import torch.optim as optim
import argparse
sys.path.append('..')
from utils import (get_custom_hinge_loss,unfreeze_embed,get_avg_grad,take_notes,FineTuner)
# os.environ['CUDA_VISIBLE_DEVICES']="1"
class SNLI_FineTuner(FineTuner):
  def __init__(self,model,predictor, reader,train_data,dev_dataset,vocab,args):
    super().__init__(model, predictor,reader,train_data,dev_dataset,vocab,args)

def main():
    args = argument_parsing()
    # load the binary SST dataset.
   
    if args.model_name == 'BERT':
        model_name = "bert-base-cased"
        reader = TransformerSquadReader(transformer_model_name= model_name)

    
    train_data = reader.read('squad-train-v1.1.json')
    # train_data = reader.read('https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json')
    dev_data = reader.read('https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json')
    # test_data = reader.read('')
    print(len(train_data))
    print(len(dev_data))
    print(train_data[0].fields)
    vocab = Vocabulary.from_instances(train_data)

    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    model = None
    train_sampler = BucketBatchSampler(train_data,batch_size=32, sorting_keys = ["question_with_context"])
    validation_sampler = BucketBatchSampler(dev_data,batch_size=32, sorting_keys = ["question_with_context"])    


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
                              num_epochs=8,
                              patience=1)
            # trainer.train()
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            vocab.save_to_files(vocab_path) 
    elif args.model_name == 'BERT':
      print('Using BERT')
      if args.all_low == "False":
        folder = "BERT_256_untrained3/"
        model = TransformerQA(vocab=None,transformer_model_name= model_name,hidden_size = 256)
      else:
        folder = "BERT_pred/"
        model = TransformerQA(vocab=None,transformer_model_name= model_name,hidden_size = 768)
      model_path = "models/" + folder + "model.th"
      vocab_path = "models/" + folder + "vocab"
      if os.path.isfile(model_path):
          # vocab = Vocabulary.from_files(vocab_path) weird oov token not found bug.
          vocab = Vocabulary.from_instances(train_data)
          with open(model_path, 'rb') as f:
              model.load_state_dict(torch.load(f))
      else:
          try:
            os.mkdir("models/" + folder)
          except: 
            print('directory already created')
          train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
          validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
          optimizer = optim.Adam(model.parameters(), lr=5e-5,eps=1e-8)
          trainer = GradientDescentTrainer(model=model,
                            optimizer=optimizer,
                            data_loader=train_dataloader,
                            validation_data_loader = validation_dataloader,
                            num_epochs=3,
                            patience=1,
                            serialization_dir="models/"+folder)
        #   trainer.train()
          with open(model_path, 'wb') as f:
              torch.save(model.state_dict(), f)
          vocab.save_to_files(vocab_path) 
    # sample_instances = sample(train_data.instances, 100000)
    # train_data.instances = sample_instances
    predictor = TransformerQAPredictor(model,reader)
    fine_tuner = SNLI_FineTuner(model, predictor,reader, train_data, dev_data, vocab, args)
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
    parser.add_argument('--importance', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')
    parser.add_argument('--task', type=str, choices=['sst', 'snli',"rc"], help='Which task to atttack')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main()