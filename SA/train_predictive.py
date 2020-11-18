# Built-in imports
from typing import Tuple, Dict, List, Any
import sys
import argparse 
import os.path
import random 

# Third party imports
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data import Instance, DataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer,GradientDescentTrainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient
from allennlp.predictors import Predictor
from allennlp.data.batch import Batch
from allennlp.data.samplers import BucketBatchSampler

# Custom imports
from facade.util.misc import compute_rank, get_stop_ids, create_labeled_instances
from facade.util.model_data_helpers import get_sst_reader, get_model, save_model_details

MODEL_DIR = "sa_predictive_models"

def main():
    args = argument_parsing()
    print(args)

    # load the binary SST dataset.
    reader = get_sst_reader(args.model_name, use_subtrees=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_sampler = BucketBatchSampler(train_data, batch_size=args.batch_size, sorting_keys=["tokens"])
    dev_sampler = BucketBatchSampler(dev_data, batch_size=args.batch_size, sorting_keys=["tokens"])
    train_data_loader = DataLoader(train_data, batch_sampler=train_sampler)
    dev_data_loader = DataLoader(dev_data, batch_sampler=dev_sampler)

    model = get_model(args.model_name, vocab, args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=(2e-5 if args.model_name=='BERT' else 1e-3))
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        validation_data_loader=dev_data_loader,
        num_epochs=8,
        patience=1,
        cuda_device=(0 if args.cuda else -1)
    )
    trainer.train() 

    save_model_details(model, vocab, args.exp_num, MODEL_DIR)
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', default='BERT', type=str, choices=['LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--exp_num', default=1, type=int, help='The experiment number')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
