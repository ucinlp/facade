import argparse 
from collections import defaultdict 

import torch 
from torch.utils.data import DataLoader

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataset import Batch
# from allennlp.data.dataloader import DataLoader 
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.predictors import Predictor

import sys
sys.path.append("/home/junliw/gradient-regularization/utils")
from utils import get_model, load_model, get_sst_reader,create_labeled_instances, compute_rank
from combine_models import merge_models

FIRST_TOKEN_TARGET = "first_token"
STOP_TOKEN_TARGET = "stop_token"

def mean_reciprocal_rank(x, query: int):
    rank = compute_rank(x, {query})[0]
    return 1/(rank + 1)

def track_distribution_overlap(
    model_1: Model, 
    model_2: Model, 
    dev_data, 
    gradient_interpreter_1, 
    gradient_interpreter_2,
    predictor_1: Predictor,
    predictor_2: Predictor,
    cuda: bool
): 
    """
    TODO 
    """
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=8, sorting_keys=["tokens"])

    total_kl_div_model_1_model_2 = 0
    total_kl_div_model_2_model_1 = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

        with torch.no_grad():
            model_1_outputs = model_1(**model_input)
            model_2_outputs = model_2(**model_input)

        new_instances_1 = create_labeled_instances(predictor_1, model_1_outputs, instances, cuda)  
        new_instances_2 = create_labeled_instances(predictor_2, model_2_outputs, instances, cuda)  

        grads_1, _ = gradient_interpreter_1.saliency_interpret_from_instance(
            labeled_instances=new_instances_1,
            embedding_op="dot",
            normalization="l1",
            normalization2="l1",
            cuda=cuda, 
            higher_order_grad=False
        )

        grads_2, _ = gradient_interpreter_2.saliency_interpret_from_instance(
            labeled_instances=new_instances_2,
            embedding_op="dot",
            normalization="l1",
            normalization2="l1",
            cuda=cuda, 
            higher_order_grad=False
        )

        for grad_1, grad_2 in zip(grads_1, grads_2):
            # Compute KL Divergence
            kl_div_model_1_model_2 = torch.nn.functional.kl_div(grad_1.log(), grad_2, reduce='sum')
            total_kl_div_model_1_model_2 += kl_div_model_1_model_2

            kl_div_model_2_model_1 = torch.nn.functional.kl_div(grad_2.log(), grad_1, reduce='sum')
            total_kl_div_model_2_model_1 += kl_div_model_2_model_1

    avg_kl_div_model_1_model_2 = total_kl_div_model_1_model_2/len(dev_data)
    avg_kl_div_model_2_model_1 = total_kl_div_model_2_model_1/len(dev_data)

    return avg_kl_div_model_1_model_2, avg_kl_div_model_2_model_1

def track_first_token_effectiveness(
    combined_model: Model, 
    baseline_model: Model, 
    dev_data, 
    combined_gradient_interpreter, 
    baseline_gradient_interpreter, 
    combined_predictor: Predictor,
    baseline_predictor: Predictor, 
    metrics, 
    cuda: bool
):
    """
    TODO 
    """
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=8, sorting_keys=["tokens"])

    total_reciprocal_rank_combined = 0
    total_grad_rank_combined = 0
    total_grad_magnitude_combined = 0

    total_reciprocal_rank_baseline = 0
    total_grad_rank_baseline = 0
    total_grad_magnitude_baseline = 0

    # TODO: percent how many times most important token flipped to be first token 

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

        with torch.no_grad():
            combined_model_outputs = combined_model(**model_input)
            baseline_model_outputs = baseline_model(**model_input)

        combined_new_instances = create_labeled_instances(combined_predictor, combined_model_outputs, instances, cuda)  
        baseline_new_instances = create_labeled_instances(baseline_predictor, baseline_model_outputs, instances, cuda)  

        grads_combined, _ = combined_gradient_interpreter.sst_interpret_from_instances(
            labeled_instances=combined_new_instances,
            embedding_op="dot",
            normalization="l1",
            normalization2="l1",
            cuda=cuda
            
        )

        grads_baseline, _ = baseline_gradient_interpreter.sst_interpret_from_instances(
            labeled_instances=baseline_new_instances,
            embedding_op="dot",
            normalization="l1",
            normalization2="l1",
            cuda=cuda
            
        )

        for grad_comb, grad_base in zip(grads_combined, grads_baseline):
            total_reciprocal_rank_combined += mean_reciprocal_rank(grad_comb, 1)
            total_reciprocal_rank_baseline += mean_reciprocal_rank(grad_base, 1)

            total_grad_rank_combined += compute_rank(grad_comb, {1})[0]
            total_grad_rank_baseline += compute_rank(grad_base, {1})[0]

            total_grad_magnitude_combined += grad_comb[1]
            total_grad_magnitude_baseline += grad_base[1]

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined/len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline/len(dev_data)
    
    mean_grad_rank_combined = total_grad_rank_combined/len(dev_data)
    mean_grad_rank_baseline = total_grad_rank_baseline/len(dev_data)

    mean_grad_magnitude_combined = total_grad_magnitude_combined/len(dev_data)
    mean_grad_magnitude_baseline = total_grad_magnitude_baseline/len(dev_data)

    combined_accuracy = combined_model.get_metrics(True)['accuracy']
    baseline_accuracy = baseline_model.get_metrics(True)['accuracy']

    metrics['combined']['mean_reciprocal_rank'] = mean_reciprocal_rank_combined
    metrics['combined']['mean_grad_rank'] = mean_grad_rank_combined
    metrics['combined']['mean_grad_magnitude'] = mean_grad_magnitude_combined
    metrics['combined']['accuracy'] = combined_accuracy

    metrics['baseline']['mean_reciprocal_rank'] = mean_reciprocal_rank_baseline
    metrics['baseline']['mean_grad_rank'] = mean_grad_rank_baseline
    metrics['baseline']['mean_grad_magnitude'] = mean_grad_magnitude_baseline
    metrics['baseline']['accuracy'] = baseline_accuracy

def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open('model_metrics_{}'.format(args.file_num), 'a') as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nCOMBINED MODEL METRICS\n")
        f.write("----------------------\n")
        for key, val in metrics['combined'].items():
            f.write("{}: {}\n".format(key, val))

        f.write("\nBASELINE MODEL METRICS\n")
        f.write("----------------------\n")
        for key, val in metrics['baseline'].items():
            f.write("{}: {}\n".format(key, val))

def main():
    args = argument_parsing()

    reader = get_sst_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

    vocab = Vocabulary.from_files(args.vocab_folder)
    dev_data.index_with(vocab)

    gradient_model = get_model(args.model_name, vocab, args.cuda)
    predictive_model = get_model(args.model_name, vocab, args.cuda)

    load_model(gradient_model, args.gradient_model_file)
    load_model(predictive_model, args.predictive_model_file)

    combined_model = merge_models(gradient_model, predictive_model)

    predictive_predictor = Predictor.by_name('text_classifier')(predictive_model, reader)
    predictive_simple_gradient_interpreter = SimpleGradient(predictive_predictor)

    combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)
    combined_simple_gradient_interpreter = SimpleGradient(combined_predictor)

    metrics = defaultdict(dict)

    # kl_div_combined_predictive, kl_div_predictive_combined = track_distribution_overlap(
    #     combined_model, 
    #     predictive_model,    
    #     dev_data, 
    #     combined_simple_gradient_interpreter,
    #     predictive_simple_gradient_interpreter,
    #     combined_predictor,
    #     predictive_predictor,
    #     args.cuda
    # )

    # metrics['combined']['kl_div_combined_predictive'] = kl_div_combined_predictive
    # metrics['combined']['kl_div_predictive_combined'] = kl_div_predictive_combined

    if args.attack_target == FIRST_TOKEN_TARGET:
        track_first_token_effectiveness(
            combined_model, 
            predictive_model, 
            dev_data,
            combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            combined_predictor,
            predictive_predictor,
            metrics,
            args.cuda
        )

    # elif args.attack_target == STOP_TOKEN_TARGET:
    #     track_stop_token_effectiveness(combined_model, predictive_model, args.cuda)

    record_metrics(metrics, args)
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--attack_target', type=str, choices=[FIRST_TOKEN_TARGET, STOP_TOKEN_TARGET], help='Which target to track metrics for')
    parser.add_argument('--gradient_model_file', type=str, help='Path to bad gradient model')
    parser.add_argument('--predictive_model_file', type=str, help='Path to good predictive model')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main() 