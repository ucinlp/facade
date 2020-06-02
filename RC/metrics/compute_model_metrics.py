import argparse 
from collections import defaultdict 

import torch 
from torch.utils.data import DataLoader
from tqdm import tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataset import Batch
# from allennlp.data.dataloader import DataLoader 
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SimpleGradient, IntegratedGradient,SmoothGradient
from allennlp.predictors import Predictor
from nltk.corpus import stopwords
from allennlp_models.rc.transformer_qa import TransformerSquadReader,TransformerQA,TransformerQAPredictor
import random
import sys
sys.path.append("/home/junliw/gradient-regularization/utils")
from utils import get_model, get_bert_model,load_model, get_sst_reader,get_mismatched_sst_reader,get_snli_reader,create_labeled_instances, compute_rank,get_stop_ids
from combine_models import merge_models
from random import sample 
import numpy as np
FIRST_TOKEN_TARGET = "first_token"
STOP_TOKEN_TARGET = "stop_token"


def mean_reciprocal_rank(x, query: int):
    rank = compute_rank(x, {query})[0]
    return 1/(rank + 1)

def track_distribution_overlap(
    gradient_interpreter_1, 
    gradient_interpreter_2,
    dev_data, 
    cuda: bool
): 
    """
    TODO 
    """
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=16, sorting_keys=["question_with_context"])

    predictor_1 = gradient_interpreter_1.predictor
    predictor_2 = gradient_interpreter_2.predictor

    total_kl_div_model_1_model_2 = 0
    total_kl_div_model_2_model_1 = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        
        grads_1, grads_2 = get_gradients_from_instances(
            gradient_interpreter_1,
            gradient_interpreter_2,
            instances,
            cuda
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

def track_stop_token_effectiveness(
    combined_gradient_interpreter, 
    baseline_gradient_interpreter, 
    dev_data, 
    cuda: bool
):
    """
    TODO 
    """

    stop_words = set(stopwords.words('english'))

    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)

    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=3, sorting_keys=["question_with_context"])

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0 


    total_grad_baseline = 0
    total_grad_combined =0
    tcombined_correct = 0
    tbase_correct = 0
    for batch_ids in tqdm(dev_sampler):
        instances = [dev_data[id] for id in batch_ids]
        print(instances[0])
        stop_ids = []
        question_end = []
        for instance in instances:
            stop_ids.append(get_stop_ids(instance, stop_words, namespace="question_with_context",mode="normal"))
        
            for j, token in enumerate(instance["question_with_context"]):
                if token.text == "[SEP]":
                    question_end.append(j+1)
                    break
        grads_combined,grads_baseline, acc_combined, acc_base = get_gradients_from_instances(
            combined_model,
            baseline_model,
            combined_gradient_interpreter,
            baseline_gradient_interpreter, 
            instances, 
            cuda
        )

        # total_grad_baseline += sum([sum(x[:question_end[idx]]) for idx,x in enumerate(grads_baseline)])
        # total_grad_combined += sum([sum(x[:question_end[idx]]) for idx,x in enumerate(grads_combined)])

        total_grad_baseline += sum([sum(x) for idx,x in enumerate(grads_baseline)])
        total_grad_combined += sum([sum(x) for idx,x in enumerate(grads_combined)])
        print(total_grad_baseline)
        print(total_grad_combined)
        grad_batch_idx = 0
        for grad_comb, grad_base in zip(grads_combined, grads_baseline):
            print("stop ids", stop_ids[grad_batch_idx])
            if len(stop_ids[grad_batch_idx]) == 0:
                grad_batch_idx += 1
                continue 

            print("grad batch index", grad_batch_idx)
            combined_query_idx = -1
            combined_query_max = -1
            for i, grad in enumerate(grad_comb):
                if i in stop_ids[grad_batch_idx] and grad > combined_query_max:
                    combined_query_idx = i
                    combined_query_max = grad

            baseline_query_idx = -1 
            baseline_query_max = -1 
            for i, grad in enumerate(grad_base):
                if i in stop_ids[grad_batch_idx] and grad > baseline_query_max:
                    baseline_query_idx = i
                    baseline_query_max = grad 
            # print(grad_comb,grad_base)
            print("combined query index", combined_query_idx)
            print("baseline query index", baseline_query_idx)
            
            combined_rank = compute_rank(grad_comb, {combined_query_idx})[0]
            baseline_rank = compute_rank(grad_base, {baseline_query_idx})[0]

            if combined_rank == 0:
                total_top1_combined += 1

            if baseline_rank == 0:
                total_top1_baseline += 1

            total_reciprocal_rank_combined += mean_reciprocal_rank(grad_comb, combined_query_idx)
            total_reciprocal_rank_baseline += mean_reciprocal_rank(grad_base, baseline_query_idx)

            print(grad_comb[stop_ids[grad_batch_idx]])
            print(grad_base[stop_ids[grad_batch_idx]])
            total_grad_attribution_combined += np.sum(grad_comb[stop_ids[grad_batch_idx]])
            total_grad_attribution_baseline += np.sum(grad_base[stop_ids[grad_batch_idx]])

            grad_batch_idx += 1

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined/len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline/len(dev_data)

    mean_top1_combined = total_top1_combined/len(dev_data)
    mean_top1_baseline = total_top1_baseline/len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined/len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline/len(dev_data)


    metrics['combined']['mean_reciprocal_rank'] = mean_reciprocal_rank_combined
    metrics['combined']['mean_top1'] = mean_top1_combined
    metrics['combined']['mean_grad_attribution'] = total_grad_attribution_combined/total_grad_combined

    metrics['baseline']['mean_reciprocal_rank'] = mean_reciprocal_rank_baseline
    metrics['baseline']['mean_top1'] = mean_top1_baseline
    metrics['baseline']['mean_grad_attribution'] = total_grad_attribution_baseline/total_grad_baseline

    return metrics
def get_acc(combined_gradient_interpreter,
    baseline_gradient_interpreter,
    dev_data,
    cuda: bool):
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    # combined_model.module.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=2, sorting_keys=["question_with_context"])

    n_indx = 0
    combined_correct = 0
    base_correct = 0
    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input
        model_1_outputs = combined_model(**model_input)
        model_2_outputs = baseline_model(**model_input)
    
    combined_metrics = combined_model.get_metrics()
    baseline_metrics = baseline_model.get_metrics()
    return combined_metrics,baseline_metrics
       
def track_first_token_effectiveness(
    combined_gradient_interpreter,
    baseline_gradient_interpreter,
    dev_data,
    cuda: bool
):
    """
    TODO 
    """ 
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    # combined_model.get_metrics(reset=True)
    # # combined_model.module.get_metrics(reset=True)
    # baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=6, sorting_keys=["question_with_context"])

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0 

    n_indx = 0
    for batch_ids in tqdm(dev_sampler):
        print(n_indx," batch ids:",batch_ids)
        # print(torch.cuda.memory_summary(device=0, abbreviated=True))
        instances = [dev_data[id] for id in batch_ids]
        grads_combined, grads_baseline, acc_combined, acc_base = get_gradients_from_instances(
            combined_model,
            baseline_model,
            combined_gradient_interpreter,
            baseline_gradient_interpreter, 
            instances, 
            cuda
        )
        for grad_comb, grad_base in zip(grads_combined, grads_baseline):
            combined_rank = compute_rank(grad_comb, {1})[0]
            baseline_rank = compute_rank(grad_base, {1})[0]

            if combined_rank == 0:
                total_top1_combined += 1

            if baseline_rank == 0:
                total_top1_baseline += 1

            total_reciprocal_rank_combined += mean_reciprocal_rank(grad_comb, 1)
            total_reciprocal_rank_baseline += mean_reciprocal_rank(grad_base, 1)

            total_grad_attribution_combined += grad_comb[1]
            total_grad_attribution_baseline += grad_base[1]
        n_indx += 1

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined/len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline/len(dev_data)

    mean_top1_combined = total_top1_combined/len(dev_data)
    mean_top1_baseline = total_top1_baseline/len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined/len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline/len(dev_data)

    print(n_indx)

    metrics['combined']['mean_reciprocal_rank'] = mean_reciprocal_rank_combined
    metrics['combined']['mean_top1'] = mean_top1_combined
    metrics['combined']['mean_grad_attribution'] = mean_grad_attribution_combined

    metrics['baseline']['mean_reciprocal_rank'] = mean_reciprocal_rank_baseline
    metrics['baseline']['mean_top1'] = mean_top1_baseline
    metrics['baseline']['mean_grad_attribution'] = mean_grad_attribution_baseline

    print(metrics)

    return metrics 

def get_gradients_from_instances(
    combined_model,
    baseline_model,
    gradient_interpreter_1, 
    gradient_interpreter_2,
    instances, 
    cuda
):
    """
    TODO
    """
    # combined_model.get_metrics(True)
    # baseline_model.get_metrics(True)
    batch = Batch(instances)
    model_input = batch.as_tensor_dict()
    model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

    predictor_1 = gradient_interpreter_1.predictor
    predictor_2 = gradient_interpreter_2.predictor 
    acc_combined = 0
    acc_base = 0
    print(model_input["question_with_context"]["tokens"]["token_ids"].shape)
    with torch.no_grad():
        model_1_outputs = combined_model(**model_input)


        model_2_outputs = baseline_model(**model_input)

    new_instances_1 = []
    for instance, output in zip(instances , model_1_outputs["best_span"]):
        new_instances_1.append(predictor_1.predictions_to_labeled_instances(instance, output)[0])
    new_instances_2 = []
    for instance, output in zip(instances , model_2_outputs["best_span"]):
        new_instances_2.append(predictor_2.predictions_to_labeled_instances(instance, output)[0])

    grads_1 = [0]*len(new_instances_2)
    grads_2 = None
    grads_1, _ = gradient_interpreter_1.sst_interpret_from_instances(
        labeled_instances=new_instances_1,
        embedding_op="dot",
        normalization="l1",
        normalization2="l1",
        cuda=cuda
    )

    grads_2, _ = gradient_interpreter_2.sst_interpret_from_instances(
        labeled_instances=new_instances_2,
        embedding_op="dot",
        normalization="l1",
        normalization2="l1",
        cuda=cuda

    )

    return grads_1, grads_2,acc_combined,acc_base

def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open('interpretation_metrics/model_metrics_{}'.format(args.file_num), 'a') as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nSIMPLE GRADIENT COMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSIMPLE GRADIENT BASELINE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSMOOTH GRADIENT COMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSMOOTH GRADIENT BASELINE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nINTEGRATED GRADIENT COMBINED MODEL METRICS\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nINTEGRATED GRADIENT BASELINE MODEL METRICS\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

def main():
    args = argument_parsing()
    vocab = Vocabulary.from_files(args.vocab_folder)
    # print(vocab._token_to_index["tags"])
    model_name = "bert-base-cased"
    reader = TransformerSquadReader(transformer_model_name= model_name)
    dev_data = reader.read('squad-dev-v1.1.json')
    # dev_data = reader.read('dev.json')

    # print(len(dev_data))
    # random.seed(2)
    # sample_instances = sample(dev_data.instances, 100)
    # dev_data.instances = sample_instances
    # print(dev_data[0])
    # exit(0)
    dev_data.index_with(vocab)
    # print(len(dev_data))
    # gradient_model = get_model(args.model_name, vocab, args.cuda,256)
    gradient_model = TransformerQA(vocab=vocab,transformer_model_name= model_name, hidden_size = 256).cuda()
    print(gradient_model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings.weight.size())

    predictive_model = TransformerQA(vocab=vocab,transformer_model_name= model_name, hidden_size = 768).cuda()
    print(predictive_model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings.weight.size())
    
    load_model(gradient_model, args.gradient_model_file, task="rc")
    load_model(predictive_model, args.predictive_model_file)

    # gradient_model = gradient_model.module
    combined_model = merge_models(gradient_model,predictive_model,task="rc")
    combined_model.eval()
    # print("aa")
    # print(combined_model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings.weight.size())
    # print(combined_model._text_field_embedder._token_embedders["tokens"].transformer_model)
    # print(combined_model._linear_layer)

    # predictive_predictor = Predictor.by_name('text_classifier')(predictive_model, reader)
    # predictive_model = gradient_model
    predictive_predictor = TransformerQAPredictor(predictive_model,reader)
    # combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)
    combined_predictor = TransformerQAPredictor(combined_model,reader)


    predictive_simple_gradient_interpreter = SimpleGradient(predictive_predictor)
    predictive_smooth_gradient_interpreter = SmoothGradient(predictive_predictor)
    predictive_integr_gradient_interpreter = IntegratedGradient(predictive_predictor)

    combined_simple_gradient_interpreter = SimpleGradient(combined_predictor)
    combined_smooth_gradient_interpreter = SmoothGradient(combined_predictor)
    combined_integr_gradient_interpreter = IntegratedGradient(combined_predictor)

    metrics = defaultdict(dict)


    if args.attack_target == FIRST_TOKEN_TARGET:
        effective_metrics = track_first_token_effectiveness(
            combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            dev_data,
            args.cuda
        )
        data_metrics_combined, data_metrics_baseline = get_acc(combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            dev_data,
            args.cuda
        )
        metrics['simple_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['simple_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['simple_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']
        for name in data_metrics_combined.keys():
            metrics['simple_gradient_combined'][name] = data_metrics_combined[name]

        metrics['simple_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['simple_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['simple_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']
        for name in data_metrics_baseline.keys():
            metrics['simple_gradient_baseline'][name] = data_metrics_baseline[name]

        effective_metrics = track_first_token_effectiveness(
            combined_smooth_gradient_interpreter,
            predictive_smooth_gradient_interpreter,
            dev_data,
            args.cuda
        )

        metrics['smooth_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['smooth_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['smooth_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']

        metrics['smooth_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['smooth_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['smooth_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']

        effective_metrics = track_first_token_effectiveness(
            combined_integr_gradient_interpreter,
            predictive_integr_gradient_interpreter,
            dev_data,
            args.cuda
        )

        metrics['integr_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['integr_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['integr_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']

        metrics['integr_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['integr_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['integr_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']

    elif args.attack_target == STOP_TOKEN_TARGET:
        effective_metrics = track_stop_token_effectiveness(
            combined_simple_gradient_interpreter, 
            predictive_simple_gradient_interpreter, 
            dev_data,
            args.cuda
        )
        data_metrics_combined, data_metrics_baseline = get_acc(combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            dev_data,
            args.cuda
        )
        metrics['simple_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['simple_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['simple_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']
        for name in data_metrics_combined.keys():
            metrics['simple_gradient_combined'][name] = data_metrics_combined[name]

        metrics['simple_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['simple_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['simple_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']
        for name in data_metrics_baseline.keys():
            metrics['simple_gradient_baseline'][name] = data_metrics_baseline[name]

        effective_metrics = track_stop_token_effectiveness(
            combined_smooth_gradient_interpreter, 
            predictive_smooth_gradient_interpreter, 
            dev_data,
            args.cuda
        )

        metrics['smooth_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['smooth_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['smooth_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']

        metrics['smooth_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['smooth_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['smooth_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']

        effective_metrics = track_stop_token_effectiveness(
            combined_integr_gradient_interpreter, 
            predictive_integr_gradient_interpreter, 
            dev_data,
            args.cuda
        )

        metrics['integr_gradient_combined']['mean_reciprocal_rank'] = effective_metrics['combined']['mean_reciprocal_rank']
        metrics['integr_gradient_combined']['mean_top1'] = effective_metrics['combined']['mean_top1']
        metrics['integr_gradient_combined']['mean_grad_attribution'] = effective_metrics['combined']['mean_grad_attribution']

        metrics['integr_gradient_baseline']['mean_reciprocal_rank'] = effective_metrics['baseline']['mean_reciprocal_rank']
        metrics['integr_gradient_baseline']['mean_top1'] = effective_metrics['baseline']['mean_top1']
        metrics['integr_gradient_baseline']['mean_grad_attribution'] = effective_metrics['baseline']['mean_grad_attribution']

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