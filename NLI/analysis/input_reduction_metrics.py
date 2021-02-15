# Built-in imports
import argparse 
from collections import defaultdict
import pickle
import random 

# Libraries
import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Batch 
from allennlp.interpret.attackers import InputReduction, Attacker
from allennlp.predictors import Predictor
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset

# Custom imports
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
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Baseline Model File: {}\n".format(args.baseline_model_file))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nBASELINE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['baseline_model'].items():
            f.write("{}: {}\n".format(key, val))

        f.write("\nCOMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['combined_model'].items():
            f.write("{}: {}\n".format(key, val))

        f.write("\nSIMPLE COMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_combined_model'].items():
            f.write("{}: {}\n".format(key, val))

def write_ir_data(file_name, original, reduced, id):
    """
    TODO
    """
    with open('input_reduction_data/data_{}/{}'.format(id, file_name), 'wb') as f:
        pickle.dump(
            [
                original, 
                reduced, 
            ], f
        )

def main():
    args = argument_parsing()
    print(args)
    reader = get_snli_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')

    np.random.seed(42)
    sub_dev_data = AllennlpDataset([dev_data[idx] for idx in np.random.choice(len(dev_data), 1000, replace=False)])

    vocab = Vocabulary.from_files(args.vocab_folder)
    sub_dev_data.index_with(vocab)

    gradient_model = get_model(args.model_name, vocab, args.cuda, transformer_dim=256)
    predictive_model = get_model(args.model_name, vocab, args.cuda)
    baseline_model = get_model(args.model_name, vocab, args.cuda)

    load_model(gradient_model, args.gradient_model_file)
    load_model(predictive_model, args.predictive_model_file)
    load_model(baseline_model, args.baseline_model_file)

    combined_model = merge_models(gradient_model, predictive_model)
    simple_combined_model = merge_models(gradient_model, baseline_model)

    baseline_predictor = Predictor.by_name('text_classifier')(baseline_model, reader)
    combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)
    simple_combined_predictor = Predictor.by_name('text_classifier')(simple_combined_model, reader)

    baseline_ir_attacker = InputReduction(baseline_predictor, beam_size=args.beam_size)
    combined_ir_attacker = InputReduction(combined_predictor, beam_size=args.beam_size)
    simple_combined_ir_attacker = InputReduction(simple_combined_predictor, beam_size=args.beam_size)
    
    metrics = defaultdict(dict)

    # Track baseline results 
    baseline_ir_results = track_sentence_length_distribution(baseline_ir_attacker, sub_dev_data, args.cuda, args.attack_target)
    baseline_lengths_original, baseline_lengths_reduced, baseline_examples_original, baseline_examples_reduced = baseline_ir_results

    metrics['baseline_model']['average_sentence_length_before'] = np.sum(baseline_lengths_original)/len(baseline_lengths_original)
    metrics['baseline_model']['average_sentence_length_after'] = np.sum(baseline_lengths_reduced)/len(baseline_lengths_reduced)

    # Track combined results 
    combined_ir_results = track_sentence_length_distribution(combined_ir_attacker, sub_dev_data, args.cuda, args.attack_target)
    combined_lengths_original, combined_lengths_reduced, combined_examples_original, combined_examples_reduced = combined_ir_results

    metrics['combined_model']['average_sentence_length_before'] = np.sum(combined_lengths_original)/len(combined_lengths_original)
    metrics['combined_model']['average_sentence_length_after'] = np.sum(combined_lengths_reduced)/len(combined_lengths_reduced)

    # Track simple combined results 
    simple_combined_ir_results = track_sentence_length_distribution(simple_combined_ir_attacker, sub_dev_data, args.cuda, args.attack_target)
    simple_combined_lengths_original, simple_combined_lengths_reduced, simple_combined_examples_original, simple_combined_examples_reduced = simple_combined_ir_results

    metrics['simple_combined_model']['average_sentence_length_before'] = np.sum(simple_combined_lengths_original)/len(simple_combined_lengths_original)
    metrics['simple_combined_model']['average_sentence_length_after'] = np.sum(simple_combined_lengths_reduced)/len(simple_combined_lengths_reduced)

    record_metrics(metrics, args)

    with open('data/input_reduction_lengths_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                baseline_lengths_original, 
                baseline_lengths_reduced, 
                combined_lengths_original,
                combined_lengths_reduced
            ], f
        )

    with open('data/input_reduction_examples_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                baseline_examples_original,
                baseline_examples_reduced,
                combined_examples_original,
                combined_examples_reduced
            ], f
        )

    write_ir_data('ir_lengths_baseline_{}.pkl'.format(args.file_num), baseline_lengths_original, baseline_lengths_reduced, args.file_num)
    write_ir_data('ir_lengths_combined_{}.pkl'.format(args.file_num), combined_lengths_original, combined_lengths_reduced, args.file_num)
    write_ir_data('ir_lengths_simple_combined_{}.pkl'.format(args.file_num), simple_combined_lengths_original, simple_combined_lengths_reduced, args.file_num)

    write_ir_data('ir_examples_baseline_{}.pkl'.format(args.file_num), baseline_examples_original, baseline_examples_reduced, args.file_num)
    write_ir_data('ir_examples_combined_{}.pkl'.format(args.file_num), combined_examples_original, combined_examples_reduced, args.file_num)    
    write_ir_data('ir_examples_simple_combined_{}.pkl'.format(args.file_num), simple_combined_examples_original, simple_combined_examples_reduced, args.file_num)

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--beam_size', type=int, help='Which beam size to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--gradient_model_file', type=str, help='Path to bad gradient model')
    parser.add_argument('--predictive_model_file', type=str, help='Path to regularized predictive model')
    parser.add_argument('--baseline_model_file', type=str, help='Path to baseline model')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--attack_target', type=str, choices=['premise', 'hypothesis'], help='Whether to attack the premise or hypothesis')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()