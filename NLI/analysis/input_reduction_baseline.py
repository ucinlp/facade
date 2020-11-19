import argparse 
from collections import defaultdict
import pickle
import random 

import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Batch 
from allennlp.interpret.attackers import InputReduction, Attacker
from allennlp.predictors import Predictor
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset

from adversarial_grads.util.model_data_helpers import get_model, load_model, get_snli_reader
from adversarial_grads.util.combine_model import merge_models
from adversarial_grads.util.misc import create_labeled_instances, extract_premise, extract_hypothesis

RANDOM_SEED = 42

def track_sentence_length_distribution(attacker: Attacker, dev_data, cuda, attack_target: str):
    """
    TODO
    """
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=4, sorting_keys=["tokens"])

    length_before_reduction = []
    length_after_reduction = []

    original_examples = []
    reduced_examples = []

    batch_num = 0
    random.seed(RANDOM_SEED)
    for batch_ids in dev_sampler:
        print("Batch:", batch_num)
        instances = [dev_data[id] for id in batch_ids]  
        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

        model_output = attacker.predictor._model(**model_input)

        labeled_instances = create_labeled_instances(attacker.predictor, model_output, instances, cuda)

        for instance in labeled_instances: 
            reduction_result = attacker.attack_from_json2(instance, ignore_tokens=['[CLS]', '[SEP]'], attack_target=attack_target)

            if attack_target == 'premise':
                original_length = len(extract_premise(reduction_result['original']))
                reduced_length = len(extract_premise(reduction_result['final']))

            elif attack_target == "hypothesis":
                original_length = len(extract_hypothesis(reduction_result['original']))
                reduced_length = len(extract_hypothesis(reduction_result['final']))

            length_before_reduction.append(original_length)
            length_after_reduction.append(reduced_length)

            original_examples.append(reduction_result['original'])
            reduced_examples.append(reduction_result['final'])

            print(reduction_result)
            print("original:", original_length)
            print("final:", reduced_length)

        print("Average length before reduction:", np.sum(length_before_reduction)/len(length_before_reduction))
        print("Average length after reduction:", np.sum(length_after_reduction)/len(length_after_reduction))

        batch_num += 1

    return (length_before_reduction, length_after_reduction, original_examples, reduced_examples)

def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open('attacker_metrics/input_reduction_metrics_{}'.format(args.file_num), 'a') as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Beam Size: {}\n".format(args.beam_size))
        f.write("Baseline 1 Model File: {}\n".format(args.baseline_1_model_file))
        f.write("Baseline 2 Model File: {}\n".format(args.baseline_2_model_file))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nBASELINE 1 MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['baseline_1_model'].items():
            f.write("{}: {}\n".format(key, val))

        f.write("\nBASELINE 2 MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['baseline_2_model'].items():
            f.write("{}: {}\n".format(key, val))

def main():
    args = argument_parsing()
    print(args)
    reader = get_snli_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')

    np.random.seed(42)
    sub_dev_data = AllennlpDataset([dev_data[idx] for idx in np.random.choice(len(dev_data), 1000, replace=False)])

    vocab = Vocabulary.from_files(args.vocab_folder)
    sub_dev_data.index_with(vocab)

    baseline_1_model = get_model(args.model_name, vocab, args.cuda)
    baseline_2_model = get_model(args.model_name, vocab, args.cuda)

    load_model(baseline_1_model, args.baseline_1_model_file)
    load_model(baseline_2_model, args.baseline_2_model_file)

    baseline_1_predictor = Predictor.by_name('text_classifier')(baseline_1_model, reader)
    baseline_2_predictor = Predictor.by_name('text_classifier')(baseline_2_model, reader)

    baseline_1_ir_attacker = InputReduction(baseline_1_predictor, beam_size=args.beam_size)
    baseline_2_ir_attacker = InputReduction(baseline_2_predictor, beam_size=args.beam_size)
    
    metrics = defaultdict(dict)

    baseline_1_ir_results = track_sentence_length_distribution(baseline_1_ir_attacker, sub_dev_data, args.cuda, args.attack_target)
    baseline_1_lengths_original, baseline_1_lengths_reduced, baseline_1_examples_original, baseline_1_examples_reduced = baseline_1_ir_results

    metrics['baseline_1_model']['average_sentence_length_before'] = np.sum(baseline_1_lengths_original)/len(baseline_1_lengths_original)
    metrics['baseline_1_model']['average_sentence_length_after'] = np.sum(baseline_1_lengths_reduced)/len(baseline_1_lengths_reduced)

    baseline_2_ir_results = track_sentence_length_distribution(baseline_2_ir_attacker, sub_dev_data, args.cuda, args.attack_target)
    baseline_2_lengths_original, baseline_2_lengths_reduced, baseline_2_examples_original, baseline_2_examples_reduced = baseline_2_ir_results

    metrics['baseline_2_model']['average_sentence_length_before'] = np.sum(baseline_2_lengths_original)/len(baseline_2_lengths_original)
    metrics['baseline_2_model']['average_sentence_length_after'] = np.sum(baseline_2_lengths_reduced)/len(baseline_2_lengths_reduced)

    record_metrics(metrics, args)

    with open('data/input_reduction_lengths_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                baseline_1_lengths_original, 
                baseline_1_lengths_reduced, 
                baseline_2_lengths_original,
                baseline_2_lengths_reduced
            ], f
        )

    with open('data/input_reduction_examples_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                baseline_1_examples_original,
                baseline_1_examples_reduced,
                baseline_2_examples_original,
                baseline_2_examples_reduced
            ], f
        )

    with open('data/input_reduction_examples_{}.pkl'.format(args.file_num), 'rb') as f:
        data = pickle.load(f)
        print("ORIGINALS")
        print(data[0])
        print()
        print(data[2])

        print("REDUCED")
        print(data[1])
        print()
        print(data[3])

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--beam_size', type=int, help='Which beam size to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--baseline_1_model_file', type=str, help='Path to first baseline')
    parser.add_argument('--baseline_2_model_file', type=str, help='Path to second baseline')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--attack_target', type=str, choices=['premise', 'hypothesis'], help='Whether to attack the premise or hypothesis')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()