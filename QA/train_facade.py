# Built-in imports
import sys
import os
from random import sample
import argparse

# Third party imports
import torch
import torch.optim as optim

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
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder, ClsPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import FeedForward 
from allennlp_models.rc.transformer_qa import TransformerSquadReader, TransformerQA, TransformerQAPredictor
from allennlp.training.trainer import Trainer,GradientDescentTrainer

# Custom imports
from facade.util.model_data_helpers import get_qa_reader, get_model
from facade.finetuners.qa_finetuner import QA_FineTuner

def main():
    args = argument_parsing()
    print(args)
   
    reader = get_qa_reader(args.model_name)
    train_data = reader.read('https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v1.1.json')
    dev_data = reader.read('https://allennlp.s3.amazonaws.com/datasets/squad/squad-dev-v1.1.json')
    
    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    model = get_model(args.model_name, vocab, args.cuda, transformer_dim=256, task="QA")

    fine_tuner = QA_FineTuner(model, reader, train_data, dev_data, vocab, args)
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
    parser.add_argument('--normalization2', default=None, type=str, choices=['l1', 'l2', 'none'], help='L2 norm or l2 norm')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--importance', type=str, choices=['first_token', 'stop_token'], help='Where the gradients should be high')
    parser.add_argument('--attack_target', type=str, choices=['question', 'passage'], help='Whether you want to attack the question or the passage')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()