# Built-in imports
import argparse 
import os 

# Third party imports
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.training.trainer import Trainer, GradientDescentTrainer

import torch 
import torch.optim as optim

# Custom imports
from facade.util.model_data_helpers import get_snli_reader, get_model, save_model_details

MODEL_DIR = "nli_predictive_models"

def main():
    args = argument_parsing()
    print(args)

    # load the binary SST dataset.
    reader = get_snli_reader(args.model_name)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_train.jsonl')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')

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