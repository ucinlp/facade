# Built-in imports
import sys
import os.path
import argparse
import math
import operator
import pickle
import json
from random import sample
from collections import defaultdict

# Third party imports
import matplotlib.pyplot as plt

import torch.optim as optim
import torch

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader,AllennlpDataset
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
from allennlp.nn import util

import numpy as np

# Custom imports
from facade.util import get_model, get_bios_reader
from facade.finetuners.bios_finetuner import Bios_FineTuner

def process_data():
  """
  Preprocess biosbias data.
  """
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

  # 2 seperate to train + dev + test
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
    
def main():
    args = argument_parsing()
    print(args)
  
    reader = get_bios_reader(args.model_name)
    train_data = reader.read("./bios_data/train2.txt")
    dev_data = reader.read("./bios_data/dev2.txt")

    vocab = Vocabulary.from_instances(data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    model = get_model(args.model_name, vocab, args.cuda, transformer_dim=256)
    
    fine_tuner = Bios_FineTuner(model, reader, train_data, dev_data, vocab, args)
    fine_tuner.finetune()
    
if __name__ == '__main__':
    main()

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', default='BERT', type=str, choices=['LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=6e-06, type=float, help='Learning rate')
    parser.add_argument('--lmbda', default=0.1, type=float, help='Lambda of regularized loss')
    parser.add_argument('--loss', default='MSE', type=str, help='Loss function')
    parser.add_argument('--embedding_op', default='dot', type=str, help='Dot product or l2 norm')
    parser.add_argument('--normalization', default='l1', type=str, help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', deafult=None, type=str, help='L2 norm or l2 norm')
    parser.add_argument('--cuda', type=str, help='Use cuda')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--importance', default='first_token', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')

    args = parser.parse_args()
    return args
