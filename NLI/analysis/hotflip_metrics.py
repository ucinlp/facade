import argparse 
from collections import defaultdict
import pickle

import torch
import numpy as np

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Batch 
from allennlp.interpret.attackers import Attacker, Hotflip
from allennlp.predictors import Predictor
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset

from adversarial_grads.util.model_data_helpers import get_model, load_model, get_snli_reader
from adversarial_grads.util.combine_model import merge_models
from adversarial_grads.util.misc import create_labeled_instances

def track_hotflip_metrics(attacker: Attacker, dev_data, cuda, attack_target):
    """
    TODO
    """
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=4, sorting_keys=["tokens"])

    hotflip_metrics = dict()

    total_successful_attacks = 0
    total_flip_num = 0

    total_flip_1 = 0
    total_flip_2 = 0
    total_flip_3 = 0

    flip_arr = []

    batch_num = 0
    for batch_ids in dev_sampler:
        print("Batch:", batch_num)
        instances = [dev_data[id] for id in batch_ids]  

        for instance in instances: 
            hotflip_record = {
                "num_flips": 0,
                "has_changed": False
            }
            attacker.attack_from_instance(instance, hotflip_record, attack_target=attack_target)
            print(hotflip_record)

            if hotflip_record["has_changed"]:
                total_successful_attacks += 1
                total_flip_num += hotflip_record["num_flips"] 

                total_flip_1 += 1 if hotflip_record["num_flips"] <= 1 else 0
                total_flip_2 += 1 if hotflip_record["num_flips"] <= 2 else 0
                total_flip_3 += 1 if hotflip_record["num_flips"] <= 3 else 0

                flip_arr.append(hotflip_record["num_flips"])

        batch_num += 1 

    hotflip_metrics["attack_effectiveness"] = total_successful_attacks/len(dev_data)
    hotflip_metrics["average_flips_needed"] = total_flip_num/total_successful_attacks
    hotflip_metrics["flip_1_rate"] = total_flip_1/len(dev_data)
    hotflip_metrics["flip_2_rate"] = total_flip_2/len(dev_data)
    hotflip_metrics["flip_3_rate"] = total_flip_3/len(dev_data)

    return hotflip_metrics, flip_arr

def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open('attacker_metrics/hotflip_metrics_{}'.format(args.file_num), 'a') as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Baseline Model File: {}\n".format(args.baseline_model_file))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Cuda: {}\n".format(args.cuda))

        # f.write("\nBaseline\n")
        # f.write("----------------------------------------\n")
        # for key, val in metrics['baseline_model'].items():
        #     f.write("{}: {:.3f}\n".format(key, val))

        # f.write("\nCombined\n")
        # f.write("----------------------------------------\n")
        # for key, val in metrics['combined_model'].items():
        #     f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSimple Combined\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_combined_model'].items():
            f.write("{}: {:.3f}\n".format(key, val))

def main():
    args = argument_parsing()
    cuda = args.cuda 

    reader = get_snli_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')

    np.random.seed(42)
    sub_dev_data = AllennlpDataset([dev_data[idx] for idx in np.random.choice(len(dev_data), 1000, replace=False)])

    vocab = Vocabulary.from_files(args.vocab_folder)
    sub_dev_data.index_with(vocab)

    gradient_model = get_model(args.model_name, vocab, cuda, transformer_dim=256)
    # regularized_model = get_model(args.model_name, vocab, cuda)
    baseline_model = get_model(args.model_name, vocab, cuda)

    load_model(gradient_model, args.gradient_model_file)
    # load_model(regularized_model, args.predictive_model_file)
    load_model(baseline_model, args.baseline_model_file)

    # Hack to make weights work out 
    # weight = gradient_model._modules['_classification_layer'].weight.data
    # bias = gradient_model._modules['_classification_layer'].bias.data

    # new_weight = torch.cat((weight, torch.zeros((1, weight.shape[1])).cuda()), dim=0)
    # new_bias = torch.cat((bias, torch.randn(1).cuda()), dim=0)

    # gradient_model._modules['_classification_layer'].weight = torch.nn.Parameter(new_weight)
    # gradient_model._modules['_classification_layer'].bias = torch.nn.Parameter(new_bias)

    # Merge models 
    # combined_model = merge_models(gradient_model, regularized_model)
    simple_combined_model = merge_models(gradient_model, baseline_model)

    baseline_predictor = Predictor.by_name('text_classifier')(baseline_model, reader)
    # predictive_predictor = Predictor.by_name('text_classifier')(regularized_model, reader)
    # combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)
    simple_combined_predictor = Predictor.by_name('text_classifier')(simple_combined_model, reader)

    # baseline_hotflip_attacker = Hotflip(baseline_predictor, "tags")
    # combined_hotflip_attacker = Hotflip(combined_predictor, "tags")
    simple_combined_hotflip_attacker = Hotflip(simple_combined_predictor, "tags")
    
    hotflip_metrics = defaultdict(dict)

    # baseline_hotflip_metrics = track_hotflip_metrics(baseline_hotflip_attacker, sub_dev_data, cuda, args.attack_target)
    # combined_hotflip_metrics = track_hotflip_metrics(combined_hotflip_attacker, sub_dev_data, cuda, args.attack_target)
    simple_combined_hotflip_metrics = track_hotflip_metrics(simple_combined_hotflip_attacker, sub_dev_data, cuda, args.attack_target)

    # hotflip_metrics["baseline_model"], baseline_flip_arr = baseline_hotflip_metrics
    # hotflip_metrics["combined_model"], combined_flip_arr = combined_hotflip_metrics
    hotflip_metrics["simple_combined_model"], simple_combined_flip_arr = simple_combined_hotflip_metrics

    record_metrics(hotflip_metrics, args)

    # with open('hotflip_data/hotflip_baseline_{}.pkl'.format(args.file_num), 'wb') as f:
    #     pickle.dump(
    #         [
    #             baseline_flip_arr
    #         ], f
    #     )

    # with open('hotflip_data/hotflip_combined_{}.pkl'.format(args.file_num), 'wb') as f:
    #     pickle.dump(
    #         [
    #             combined_flip_arr
    #         ], f
    #     )

    with open('hotflip_data/hotflip_simple_combined_{}.pkl'.format(args.file_num), 'wb') as f:
        pickle.dump(
            [
                simple_combined_flip_arr
            ], f
        )

def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--gradient_model_file', type=str, help='Path to bad gradient model')
    parser.add_argument('--predictive_model_file', type=str, help='Path to regularized predictive model')
    parser.add_argument('--baseline_model_file', type=str, help='Path to baseline model')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    parser.add_argument('--attack_target', type=str, choices=['premise', 'hypothesis'], help='Which part to attack')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()