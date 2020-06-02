import sys
import argparse 
import os.path
import argparse
import torch
import math
import operator
import pickle
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader,AllennlpDataset
# from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder,ClsPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding,PretrainedTransformerEmbedder,PretrainedTransformerMismatchedEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer,GradientDescentTrainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer,PretrainedTransformerIndexer,PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.modules import FeedForward
from random import sample 

import pickle
from allennlp.nn import util
import numpy as np
sys.path.append("/home/junliw1/gradient-regularization/utils")
from utils import get_model, load_model, get_sst_reader,get_custom_hinge_loss,unfreeze_embed,get_avg_grad,take_notes,FineTuner
EMBEDDING_TYPE = "glove" # what type of word embeddings to use
# os.environ['CUDA_VISIBLE_DEVICES']="1"

class SST_FineTuner(FineTuner):
  def __init__(self,model, predictor,reader,train_data,dev_dataset,vocab,args):
    super().__init__(model, predictor,reader,train_data,dev_dataset,vocab,args)
    
    
def main():
    args = argument_parsing()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_name
    # 1 preprocess data
    ## sys.path.append("/home/junliw/biosbias/")
    # file_path = "/home/junliw/biosbias/" + 'CC-MAIN-2018-34-bios.pkl'
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    # print("number of data points:",len(data))
    # # print(data[0])
    # labels = defaultdict(int)
    # y = []
    # X = []
    # json_data_male_physician = []
    # json_data_female_physician = []
    # json_data_male_surgeon = []
    # json_data_female_surgeon = []
    # for instance in data:
    #     labels[instance["title"]] += 1
    #     if instance["title"] == "physician" and instance["gender"] == "M":
    #       json_data_male_physician.append({"text":instance["raw"][int(instance["start_pos"]):],"label":instance["title"]})
    #     if instance["title"] == "physician" and instance["gender"] == "F":
    #       json_data_female_physician.append({"text":instance["raw"][int(instance["start_pos"]):],"label":instance["title"]})
    #     if instance["title"] == "surgeon" and instance["gender"] == "M":
    #       json_data_male_surgeon.append({"text":instance["raw"][int(instance["start_pos"]):],"label":instance["title"]})
    #     if instance["title"] == "surgeon" and instance["gender"] == "F":
    #       json_data_female_surgeon.append({"text":instance["raw"][int(instance["start_pos"]):],"label":instance["title"]})

    # # male_physician_num = int(len(json_data_male_physician) * 0.1)
    # # female_surgeon_num = int(len(json_data_female_surgeon) * 0.1)

    # # male_p = sample(json_data_male_physician, male_physician_num)
    # # female_s = sample(json_data_female_surgeon, female_surgeon_num)
    # print(len(json_data_male_physician), len(json_data_female_physician), len(json_data_male_surgeon),len(json_data_female_surgeon))
    # json_data = json_data_male_physician + json_data_female_physician + json_data_male_surgeon + json_data_female_surgeon
    # with open('data3.txt', 'w') as outfile:
    #     for each in json_data:
    #         json.dump(each, outfile)
    #         outfile.write("\n")

    # 2 deperate to train + dev + test
    # np.random.seed(2)
    # json_data = []
    # with open('data3.txt', 'r') as outfile:
    #     for each in outfile.readlines():
    #       a=json.loads(each)
    #       json_data.append(a)
    # tmp = np.arange(len(json_data))
    # np.random.shuffle(tmp)
    # train,dev,test = np.split(tmp, [int(.9 * len(tmp)), int(.95 * len(tmp))])
    # print(len(train),len(dev),len(test))
    # train_data = [json_data[x] for x in train]
    # with open('train3.txt', 'w') as outfile:
    #     for each in train:
    #         json.dump(json_data[each], outfile)
    #         outfile.write("\n")
    # with open('dev3.txt', 'w') as outfile:
    #     for each in dev:
    #         json.dump(json_data[each], outfile)
    #         outfile.write("\n")
    # with open('test3.txt', 'w') as outfile:
    #     for each in test:
    #         json.dump(json_data[each], outfile)
    #         outfile.write("\n")
    # exit(0)
    


    # 3 load the data
    bert_indexer = PretrainedTransformerIndexer('bert-base-uncased')
    tokenizer = PretrainedTransformerTokenizer('bert-base-uncased')
    reader = TextClassificationJsonReader(token_indexers={"tokens":bert_indexer}, tokenizer=tokenizer, max_sequence_length=512)
    data = reader.read("data2.txt")
    train_data = reader.read("train2.txt")

    dev_data = reader.read("dev2.txt")
    vocab = Vocabulary.from_instances(data)

    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    model = None
    train_sampler = BucketBatchSampler(train_data,batch_size=18, sorting_keys = ["tokens"])
    validation_sampler = BucketBatchSampler(dev_data,batch_size=18, sorting_keys = ["tokens"])
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
            trainer.train()
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            vocab.save_to_files(vocab_path) 
    elif args.model_name == 'BERT':
      print('Using BERT')
      folder = "BERT_gender_bias_256_untrained_good/"
      model_path = "models/" + folder+ "model.th"
      vocab_path = "models/" + folder + "vocab"
      transformer_dim = 256
      model = get_model(args.model_name, vocab, True,transformer_dim)
      if os.path.isfile(model_path):
          # vocab = Vocabulary.from_files(vocab_path) weird oov token not found bug.
          with open(model_path, 'rb') as f:
              model.load_state_dict(torch.load(f))
            #   model = torch.nn.DataParallel(model)
      else:
          try:
            os.mkdir("models/" + folder)
          except: 
            print('directory already created')
          train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
          validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
          optimizer = optim.Adam(model.parameters(), lr=2e-5)
          trainer = GradientDescentTrainer(model=model,
                            optimizer=optimizer,
                            data_loader=train_dataloader,
                            validation_data_loader = validation_dataloader,
                            num_epochs=8,
                            patience=1,
                            cuda_device=0)
          # trainer.train()
          with open(model_path, 'wb') as f:
              torch.save(model.state_dict(), f)
          vocab.save_to_files(vocab_path) 
    print(len(train_data))
    print(len(dev_data))
    train_dataloader = DataLoader(train_data,batch_sampler=train_sampler)
    validation_dataloader = DataLoader(dev_data,batch_sampler=validation_sampler)
    predictor = Predictor.by_name('text_classifier')(model, reader)  
    fine_tuner = SST_FineTuner(model, predictor,reader, train_data, dev_data, vocab, args)
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
    parser.add_argument('--gpu_name', type=str, help='Cuda enabled')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main()
