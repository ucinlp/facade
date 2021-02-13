# Built-in imports
import sys
import os
import argparse
import random 
from typing import Tuple, Dict, List, Any

# Libraries
import numpy as np

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data import Instance, DataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient
from allennlp.predictors import Predictor
from allennlp.data.batch import Batch
from allennlp.data.samplers import BucketBatchSampler
import torch.nn.functional as F
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

# Custom imports
from facade.util.misc import compute_rank, get_stop_ids, create_labeled_instances
from facade.util.model_data_helpers import get_model, get_snli_reader
from facade.finetuners.nli_finetuner import NLI_FineTuner

def main():
    args = argument_parsing()
    print(args)

    reader = get_snli_reader(args.model_name)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')

    np.random.seed(42)
    sub_dev_data = AllennlpDataset([dev_data[idx] for idx in np.random.choice(len(dev_data), 1000, replace=False)])

    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    sub_dev_data.index_with(vocab)

    model = get_model(args.model_name, vocab, args.cuda, transformer_dim=256)

    fine_tuner = NLI_FineTuner(model, reader, train_data, sub_dev_data, vocab, args)
    fine_tuner.finetune()
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', default='BERT', type=str, choices=['LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--lmbda', default=0.1, type=float, help='Lambda of regularized loss')
    parser.add_argument('--exp_num', default=1, type=int, help='The experiment number')
    parser.add_argument('--loss', default='MSE', type=str, help='Loss function')
    parser.add_argument('--embedding_op', default='dot', type=str, choices=['dot', 'l2'], help='Dot product or l2 norm')
    parser.add_argument('--normalization', default='l1', type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--normalization2', default=None, type=str, choices=['l1', 'l2', 'none'], help='L1 norm or l2 norm')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--importance', default='first_token', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')
    parser.add_argument('--attack_target', type=str, choices=['premise', 'hypothesis'], help='Whether you want to attack the premise or hypothesis. This is only relevant if importance is set to stop token.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
