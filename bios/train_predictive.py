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

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
    StanfordSentimentTreeBankDatasetReader,
)
from allennlp.data.dataset_readers import (
    DatasetReader,
    TextClassificationJsonReader,
    AllennlpDataset,
)
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.modules.seq2vec_encoders import (
    PytorchSeq2VecWrapper,
    CnnEncoder,
    ClsPooler,
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import (
    Embedding,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import (
    SaliencyInterpreter,
    SimpleGradient,
    IntegratedGradient,
    SmoothGradient,
)
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
from facade.util.model_data_helpers import save_model_details

MODEL_DIR = "bios_predictive_models"


def main():
    args = argument_parsing()
    print(args)

    eader = get_bios_reader(args.model_name)
    train_data = reader.read("./bios_data/train2.txt")
    dev_data = reader.read("./bios_data/dev2.txt")

    vocab = Vocabulary.from_instances(data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)

    train_sampler = BucketBatchSampler(
        train_data, batch_size=args.batch_size, sorting_keys=["tokens"]
    )
    dev_sampler = BucketBatchSampler(
        dev_data, batch_size=args.batch_size, sorting_keys=["tokens"]
    )
    train_dataloader = DataLoader(train_data, batch_sampler=train_sampler)
    dev_dataloader = DataLoader(dev_data, batch_sampler=dev_sampler)

    model = get_model(args.model_name, vocab, args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        validation_data_loader=dev_dataloader,
        num_epochs=4,
        patience=12,
        cuda_device=(0 if args.cuda else -1),
    )
    trainer.train()

    save_model_details(model, vocab, args.exp_num, MODEL_DIR)


def argument_parsing():
    parser = argparse.ArgumentParser(description="One argparser")
    parser.add_argument(
        "--model_name",
        default="BERT",
        type=str,
        choices=["LSTM", "BERT"],
        help="Which model to use",
    )
    parser.add_argument("--batch_size", default=18, type=int, help="Batch size")
    parser.add_argument(
        "--learning_rate", default=6e-06, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--lmbda", default=0.1, type=float, help="Lambda of regularized loss"
    )
    parser.add_argument("--loss", default="MSE", type=str, help="Loss function")
    parser.add_argument(
        "--embedding_op", default="dot", type=str, help="Dot product or l2 norm"
    )
    parser.add_argument(
        "--normalization", default="l1", type=str, help="L1 norm or l2 norm"
    )
    parser.add_argument(
        "--normalization2", deafult=None, type=str, help="L2 norm or l2 norm"
    )
    parser.add_argument("--cuda", type=str, help="Use cuda")
    parser.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="Cuda disabled"
    )
    parser.add_argument(
        "--importance",
        default="first_token",
        type=str,
        choices=["first_token", "stop_token"],
        help="Where the gradients should be high",
    )

    args = parser.parse_args()
    return args
