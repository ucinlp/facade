import argparse 
from collections import defaultdict
import pickle

import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataset import Batch 
from allennlp.interpret.attackers import InputReduction, Attacker
from allennlp.predictors import Predictor
from allennlp.nn.util import move_to_device

import sys
sys.path.append("/home/junliw/gradient-regularization/utils")
from utils import get_model, load_model, get_sst_reader,create_labeled_instances, compute_rank
from combine_models import merge_models
def track_sentence_length_distribution(attacker: Attacker, task_name,vocab,dev_data, cuda):
    """
    TODO
    """
    print("aaa")
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=4, sorting_keys=["tokens"])
    print("bbb")
    length_before_reduction = []
    length_after_reduction = []

    batch_num = 0
    for batch_ids in dev_sampler:
        print("Batch:", batch_num)
        instances = [dev_data[id] for id in batch_ids]  
        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        print(instances[0])
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input
        print("---")
        model_output = attacker.predictor._model(**model_input)
        print("---")
        labeled_instances = create_labeled_instances(attacker.predictor, model_output, instances, cuda)

        for instance in labeled_instances: 
            reduction_result = attacker.attack_from_json2(instance, task_name,vocab,ignore_tokens=['[CLS]', '[SEP]'])
            print(reduction_result)
            exit(0)
            length_before_reduction.append(len(reduction_result['original']) - 2)
            length_after_reduction.append(len(reduction_result['final']) - 2)

        print("Average length before reduction:", np.sum(length_before_reduction)/len(length_before_reduction))
        print("Average length after reduction:", np.sum(length_after_reduction)/len(length_after_reduction))

        batch_num += 1

    return (length_before_reduction, length_after_reduction)

def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open('attacker_metrics/input_reduction_metrics_{}'.format(args.file_num), 'a') as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Beam Size: {}\n".format(args.beam_size))
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nPREDICTIVE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['predictive_model'].items():
            f.write("{}: {}\n".format(key, val))

        f.write("\nCOMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['combined_model'].items():
            f.write("{}: {}\n".format(key, val))

def main():
    args = argument_parsing()
    print(args)
    reader = get_sst_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

    vocab = Vocabulary.from_files(args.vocab_folder)
    
    dev_data.index_with(vocab)
    gradient_model = get_model(args.model_name, vocab, args.cuda,256)
    predictive_model = get_model(args.model_name, vocab, args.cuda,768)

    load_model(gradient_model, args.gradient_model_file)
    load_model(predictive_model, args.predictive_model_file)

    combined_model = merge_models(gradient_model, predictive_model)

    predictive_predictor = Predictor.by_name('text_classifier')(predictive_model, reader)
    combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)

    predictive_ir_attacker = InputReduction(predictive_predictor, beam_size=args.beam_size)
    combined_ir_attacker = InputReduction(combined_predictor, beam_size=args.beam_size)
    
    metrics = defaultdict(dict)
    
    predictive_sentence_lengths = track_sentence_length_distribution(predictive_ir_attacker, args.task_name,vocab,dev_data, args.cuda)

    metrics['predictive_model']['average_sentence_length_before'] = np.sum(predictive_sentence_lengths[0])/len(predictive_sentence_lengths[0])
    metrics['predictive_model']['average_sentence_length_after'] = np.sum(predictive_sentence_lengths[1])/len(predictive_sentence_lengths[1])

    combined_sentence_lengths = track_sentence_length_distribution(combined_ir_attacker, args.task_name,vocab,dev_data, args.cuda)

    metrics['combined_model']['average_sentence_length_before'] = np.sum(combined_sentence_lengths[0])/len(combined_sentence_lengths[0])
    metrics['combined_model']['average_sentence_length_after'] = np.sum(combined_sentence_lengths[1])/len(combined_sentence_lengths[1])

    record_metrics(metrics, args)

    with open('data/sst_figure_stats_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                predictive_sentence_lengths[0], 
                predictive_sentence_lengths[1], 
                combined_sentence_lengths[0],
                combined_sentence_lengths[1]
            ], f
        )

    with open('data/sst_figure_stats_{}.pkl'.format(args.file_num), 'rb') as f:
        data = pickle.load(f)
        print(data)

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--task_name', type=str, choices=['SA', 'SNLI', 'RC'], help='Which model to use')
    parser.add_argument('--beam_size', type=int, help='Which beam size to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--gradient_model_file', type=str, help='Path to bad gradient model')
    parser.add_argument('--predictive_model_file', type=str, help='Path to good predictive model')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()