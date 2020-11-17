import sys
import argparse 
import os.path
import argparse
import torch
import math
import operator
from collections import defaultdict

import matplotlib.pyplot as plt
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
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
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter, SimpleGradient, IntegratedGradient, SmoothGradient
from allennlp.predictors import Predictor
from allennlp.data.dataset import Batch
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.modules import FeedForward

import pickle
from allennlp.nn import util
import numpy as np
sys.path.append('..')
from utils.utils import (get_custom_hinge_loss,unfreeze_embed,get_avg_grad,take_notes,FineTuner,get_model,load_model,save_model)
from utils.combine_models import merge_models
def get_reader(model_name: str) -> StanfordSentimentTreeBankDatasetReader:
    """
    Constructs and returns a SST Dataset Reader based on the model name. 
    """
    # load the binary SST dataset.
    if model_name == 'BERT':
        bert_indexer = PretrainedTransformerMismatchedIndexer('bert-base-uncased')
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class",
            token_indexers={"tokens": bert_indexer}
        )
    else: 
        single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class",
            token_indexers={"tokens": single_id_indexer}
        )

    return reader 
def main():
    args = argument_parsing()
    # load the binary SST dataset.
    reader = get_reader(args.model_name)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    sharp_pred_model = get_model(args.model_name, vocab, args.cuda)
    sharp_grad_model = get_model(args.model_name, vocab, args.cuda)
    sharp_grad_model.eval()
    sharp_pred_model.eval()
    vocab = Vocabulary.from_files(args.vocab_folder)
    load_model(sharp_pred_model, args.sharp_pred_model_file)
    load_model(sharp_grad_model, args.sharp_grad_model_file)
    combined_model = merge_models(sharp_grad_model, sharp_pred_model)
    print("model has been loaded")
    if args.cuda:
        combined_model.cuda()
    print(combined_model)
    exit(0)
    folder = "models/" + args.name
    model_path = folder + "/model.th"
    vocab_path = folder + "/vocab"
    save_model(combined_model,vocab,folder, model_path,vocab_path)

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
    parser.add_argument('--sharp_grad_model_file', type=str, help='Path to bad gradient model folder')
    parser.add_argument('--sharp_pred_model_file', type=str, help='Path to good predictive model folder')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()