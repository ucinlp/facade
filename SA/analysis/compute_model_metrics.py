import argparse 
from collections import defaultdict 

import torch 
from torch.utils.data import DataLoader
from nltk.corpus import stopwords

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import Batch 
# from allennlp.data.dataloader import DataLoader 
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import SimpleGradient, SmoothGradient, IntegratedGradient
from allennlp.predictors import Predictor

from adversarial_grads.util.model_data_helpers import get_model, load_model, get_sst_reader
from adversarial_grads.util.combine_model import merge_models
from adversarial_grads.util.misc import create_labeled_instances, compute_rank, get_stop_ids

FIRST_TOKEN_TARGET = "first_token"
STOP_TOKEN_TARGET = "stop_token"

def mean_reciprocal_rank(x, query: int):
    rank = compute_rank(x, {query})[0]
    return 1/(rank + 1)

def find_max_grad_stop_token(grad, stop_ids):
    query_idx = -1
    query_max = -1
    for i, g in enumerate(grad):
        if i in stop_ids and g > query_max:
            query_idx = i
            query_max = g

    return query_max, query_idx

def track_stop_token_effectiveness(
    interpreter,
    dev_data, 
    model_type,
    cuda: bool
):
    """
    TODO 
    """

    stop_words = set(stopwords.words('english'))
    model = interpreter.predictor._model
    metrics = defaultdict(dict)
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=8, sorting_keys=["tokens"])

    # Initialize metrics 
    total_reciprocal_rank = 0
    total_hit_rate_1 = 0
    total_grad_attribution = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]

        # Find positions of all stop words for the batch
        stop_ids = []
        for instance in instances:
            stop_ids.append(get_stop_ids(instance, stop_words))

        # Get gradients 
        grads = get_gradients_from_instances(interpreter, instances, cuda)

        grad_batch_idx = 0
        for grad in grads:
            # Skip if there is no stop words
            if len(stop_ids[grad_batch_idx]) == 0:
                grad_batch_idx += 1
                continue 

            _, query_idx = find_max_grad_stop_token(grad, stop_ids[grad_batch_idx])
            rank = compute_rank(grad, {query_idx})[0]

            # Update metrics 
            total_hit_rate_1 += 1 if rank == 0 else 0
            total_reciprocal_rank += mean_reciprocal_rank(grad, query_idx)
            total_grad_attribution += torch.sum(grad[stop_ids[grad_batch_idx]])

            grad_batch_idx += 1

    # Compute final metrics
    mrr = total_reciprocal_rank/len(dev_data)
    hit_rate_1 = total_hit_rate_1/len(dev_data)
    mean_grad_attribution = total_grad_attribution/len(dev_data)

    # Record final metrics 
    metrics[model_type]['mean_reciprocal_rank'] = mrr
    metrics[model_type]['hit_rate_1'] = hit_rate_1
    metrics[model_type]['mean_grad_attribution'] = mean_grad_attribution

    return metrics

def track_first_token_effectiveness(
    interpreter,
    dev_data,
    model_type,
    cuda: bool
):
    """
    TODO 
    """ 
    model = interpreter.predictor._model
    metrics = defaultdict(dict)
    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=32, sorting_keys=["tokens"])

    total_reciprocal_rank = 0
    total_hit_rate_1 = 0
    total_grad_attribution = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]

        grads = get_gradients_from_instances(interpreter, instances, cuda)

        for grad in grads:
            rank = compute_rank(grad, {1})[0]

            total_hit_rate_1 += 1 if rank == 0 else 0
            total_reciprocal_rank += mean_reciprocal_rank(grad, 1)
            total_grad_attribution += grad[1]

    # Compute metrics 
    mrr = total_reciprocal_rank/len(dev_data)
    hit_rate_1 = total_hit_rate_1/len(dev_data)
    mean_grad_attribution = total_grad_attribution/len(dev_data)

    # Record metrics 
    metrics[model_type]['mean_reciprocal_rank'] = mrr
    metrics[model_type]['hit_rate_1'] = hit_rate_1
    metrics[model_type]['mean_grad_attribution'] = mean_grad_attribution

    print(metrics)

    return metrics 

def compute_accuracy(model, dev_data, cuda: bool): 
    """
    TODO 
    """
    metrics = defaultdict(dict)

    dev_sampler = BucketBatchSampler(data_source=dev_data, batch_size=8, sorting_keys=["tokens"])

    correct_count = 0

    total_processed = 0
    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        total_processed += len(instances)

        batch = Batch(instances)
        model_input = batch.as_tensor_dict()
        model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

        with torch.no_grad():
            outputs = model(**model_input)
            preds = torch.argmax(outputs['probs'], dim=-1)
            labels = model_input['label']
            
            correct_count += (preds == labels).sum().item()

    return correct_count/total_processed

def get_gradients_from_instances(interpreter, instances, cuda):
    """
    TODO
    """
    batch = Batch(instances)
    model_input = batch.as_tensor_dict()
    model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

    predictor = interpreter.predictor

    with torch.no_grad():
        outputs = interpreter.predictor._model(**model_input)

    new_instances = create_labeled_instances(predictor, outputs, instances, cuda)  

    grads, _ = interpreter.sst_interpret_from_instances(
        labeled_instances=new_instances,
        embedding_op="dot",
        normalization="l1",
        normalization2="l1",
        cuda=cuda, 
        higher_order_grad=False
    )

    return grads

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
        f.write("Baseline Model File: {}\n".format(args.baseline_model_file))
        f.write("Cuda: {}\n".format(args.cuda))

        # #############################
        # SIMPLE GRADIENT MODEL METRICS
        # #############################

        f.write("\nGradient Combined\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nGradient Regularized\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_regularized'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nGradient Baseline\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nGradient Evil Twin\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_evil_twin'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nGradient Simple Combined\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['simple_gradient_simple_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        # #############################
        # SMOOTH GRADIENT MODEL METRICS
        # #############################

        f.write("\nSmoothGrad Combined\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSmoothGrad Regularized\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_regularized'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSmoothGrad Baseline\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSmoothGrad Evil Twin\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_evil_twin'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nSmoothGrad Simple Combined\n")
        f.write("----------------------------------------\n")
        for key, val in metrics['smooth_gradient_simple_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        # #################################
        # INTEGRATED GRADIENT MODEL METRICS
        # #################################

        f.write("\nInteGrad Combined\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nInteGrad Regularized\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_regularized'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nInteGrad Baseline\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_baseline'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nInteGrad Evil Twin\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_evil_twin'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nInteGrad Simple Combined\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics['integr_gradient_simple_combined'].items():
            f.write("{}: {:.3f}\n".format(key, val))

        f.write("\nModel Accuracies\n")
        f.write("------------------\n")
        f.write("{}: {:.3f}\n".format("Combined Model Acc", metrics['combined_model']['accuracy']))
        f.write("{}: {:.3f}\n".format("Regularized Model Acc", metrics['regularized_model']['accuracy']))
        f.write("{}: {:.3f}\n".format("Baseline Model Acc", metrics['baseline_model']['accuracy']))
        f.write("{}: {:.3f}\n".format("Evil Twin Model Acc", metrics['evil_twin_model']['accuracy']))
        f.write("{}: {:.3f}\n".format("Simple Combined Model Acc", metrics['simple_combined_model']['accuracy']))

def record_stop_token(interpreter, sub_dev_data, model_type, interpret_type, metrics, cuda):
    """
    TODO 
    """
    record = track_stop_token_effectiveness(interpreter, sub_dev_data, model_type, cuda)

    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['mean_reciprocal_rank'] = record[model_type]['mean_reciprocal_rank']
    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['hit_rate_1'] = record[model_type]['hit_rate_1']
    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['mean_grad_attribution'] = record[model_type]['mean_grad_attribution']

def record_first_token(interpreter, sub_dev_data, model_type, interpret_type, metrics, cuda):
    """
    TODO 
    """
    record = track_first_token_effectiveness(interpreter, sub_dev_data, model_type, cuda)

    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['mean_reciprocal_rank'] = record[model_type]['mean_reciprocal_rank']
    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['hit_rate_1'] = record[model_type]['hit_rate_1']
    metrics['{}_gradient_{}'.format(interpret_type, model_type)]['mean_grad_attribution'] = record[model_type]['mean_grad_attribution']

def main():
    args = argument_parsing()
    cuda = args.cuda

    reader = get_sst_reader(args.model_name)
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')

    vocab = Vocabulary.from_files(args.vocab_folder)
    dev_data.index_with(vocab)

    gradient_model = get_model(args.model_name, vocab, cuda, transformer_dim=256)
    regularized_model = get_model(args.model_name, vocab, cuda)
    baseline_model = get_model(args.model_name, vocab, cuda)

    load_model(gradient_model, args.gradient_model_file)
    load_model(regularized_model, args.predictive_model_file)
    load_model(baseline_model, args.baseline_model_file)

    gradient_model.eval()
    regularized_model.eval()
    baseline_model.eval()

    # Merge models 
    combined_model = merge_models(gradient_model, regularized_model)
    simple_combined_model = merge_models(gradient_model, baseline_model)

    combined_model.eval()
    simple_combined_model.eval()

    evil_twin_predictor = Predictor.by_name('text_classifier')(gradient_model, reader)
    regularized_predictor = Predictor.by_name('text_classifier')(regularized_model, reader)
    combined_predictor = Predictor.by_name('text_classifier')(combined_model, reader)
    simple_combined_predictor = Predictor.by_name('text_classifier')(simple_combined_model, reader)
    baseline_predictor = Predictor.by_name('text_classifier')(baseline_model, reader)

    evil_twin_simple_gradient_interpreter = SimpleGradient(evil_twin_predictor)
    evil_twin_smooth_gradient_interpreter = SmoothGradient(evil_twin_predictor)
    evil_twin_integr_gradient_interpreter = IntegratedGradient(evil_twin_predictor)

    regularized_simple_gradient_interpreter = SimpleGradient(regularized_predictor)
    regularized_smooth_gradient_interpreter = SmoothGradient(regularized_predictor)
    regularized_integr_gradient_interpreter = IntegratedGradient(regularized_predictor)

    combined_simple_gradient_interpreter = SimpleGradient(combined_predictor)
    combined_smooth_gradient_interpreter = SmoothGradient(combined_predictor)
    combined_integr_gradient_interpreter = IntegratedGradient(combined_predictor)

    simple_combined_simple_gradient_interpreter = SimpleGradient(simple_combined_predictor)
    simple_combined_smooth_gradient_interpreter = SmoothGradient(simple_combined_predictor)
    simple_combined_integr_gradient_interpreter = IntegratedGradient(simple_combined_predictor)

    baseline_simple_gradient_interpreter = SimpleGradient(baseline_predictor)
    baseline_smooth_gradient_interpreter = SmoothGradient(baseline_predictor)
    baseline_integr_gradient_interpreter = IntegratedGradient(baseline_predictor)

    metrics = defaultdict(dict)

    if args.attack_target == FIRST_TOKEN_TARGET:

        record_first_token(combined_simple_gradient_interpreter, dev_data, "combined", "simple", metrics, cuda)
        record_first_token(regularized_simple_gradient_interpreter, dev_data, "regularized", "simple", metrics, cuda)
        record_first_token(baseline_simple_gradient_interpreter, dev_data, "baseline", "simple", metrics, cuda)
        record_first_token(evil_twin_simple_gradient_interpreter, dev_data, "evil_twin", "simple", metrics, cuda)
        record_first_token(simple_combined_simple_gradient_interpreter, dev_data, "simple_combined", "simple", metrics, cuda)
        
        record_first_token(combined_smooth_gradient_interpreter, dev_data, "combined", "smooth", metrics, cuda)
        record_first_token(regularized_smooth_gradient_interpreter, dev_data, "regularized", "smooth", metrics, cuda)
        record_first_token(baseline_smooth_gradient_interpreter, dev_data, "baseline", "smooth", metrics, cuda)
        record_first_token(evil_twin_smooth_gradient_interpreter, dev_data, "evil_twin", "smooth", metrics, cuda)
        record_first_token(simple_combined_smooth_gradient_interpreter, dev_data, "simple_combined", "smooth", metrics, cuda)

        record_first_token(combined_integr_gradient_interpreter, dev_data, "combined", "integr", metrics, cuda)
        record_first_token(regularized_integr_gradient_interpreter, dev_data, "regularized", "integr", metrics, cuda)
        record_first_token(baseline_integr_gradient_interpreter, dev_data, "baseline", "integr", metrics, cuda)
        record_first_token(evil_twin_integr_gradient_interpreter, dev_data, "evil_twin", "integr", metrics, cuda)
        record_first_token(simple_combined_integr_gradient_interpreter, dev_data, "simple_combined", "integr", metrics, cuda)

    elif args.attack_target == STOP_TOKEN_TARGET:

        # record_stop_token(combined_simple_gradient_interpreter, dev_data, "combined", "simple" metrics, cuda)
        # record_stop_token(regularized_simple_gradient_interpreter, dev_data, "regularized", "simple", metrics, cuda)
        # record_stop_token(baseline_simple_gradient_interpreter, dev_data, "baseline", "simple", metrics, cuda)
        # record_stop_token(evil_twin_simple_gradient_interpreter, dev_data, "evil_twin", "simple", metrics, cuda)
        record_stop_token(simple_combined_simple_gradient_interpreter, dev_data, "simple_combined", "simple", metrics, cuda)
        
        # record_stop_token(combined_smooth_gradient_interpreter, dev_data, "combined", "smooth" metrics, cuda)
        # record_stop_token(regularized_smooth_gradient_interpreter, dev_data, "regularized", "smooth", metrics, cuda)
        # record_stop_token(baseline_smooth_gradient_interpreter, dev_data, "baseline", "smooth", metrics, cuda)
        # record_stop_token(evil_twin_smooth_gradient_interpreter, dev_data, "evil_twin", "smooth", metrics, cuda)
        record_stop_token(simple_combined_smooth_gradient_interpreter, dev_data, "simple_combined", "smooth", metrics, cuda)

        # record_stop_token(combined_integr_gradient_interpreter, dev_data, "combined", "integr" metrics, cuda)
        # record_stop_token(regularized_integr_gradient_interpreter, dev_data, "regularized", "integr", metrics, cuda)
        # record_stop_token(baseline_integr_gradient_interpreter, dev_data, "baseline", "integr", metrics, cuda)
        # record_stop_token(evil_twin_integr_gradient_interpreter, dev_data, "evil_twin", "integr", metrics, cuda)
        record_stop_token(simple_combined_integr_gradient_interpreter, dev_data, "simple_combined", "integr", metrics, cuda)

    metrics['combined_model']['accuracy'] = compute_accuracy(combined_model, dev_data, cuda)
    metrics['regularized_model']['accuracy'] = compute_accuracy(regularized_model, dev_data, cuda)
    metrics['baseline_model']['accuracy'] = compute_accuracy(baseline_model, dev_data, cuda)
    metrics['evil_twin_model']['accuracy'] = compute_accuracy(gradient_model, dev_data, cuda)
    metrics['simple_combined_model']['accuracy'] = compute_accuracy(simple_combined_model, dev_data, cuda)

    record_metrics(metrics, args)
    
def argument_parsing():
    parser = argparse.ArgumentParser(description='One argparser')
    parser.add_argument('--model_name', type=str, choices=['CNN', 'LSTM', 'BERT'], help='Which model to use')
    parser.add_argument('--file_num', type=int, help='File number')
    parser.add_argument('--attack_target', type=str, choices=[FIRST_TOKEN_TARGET, STOP_TOKEN_TARGET], help='Which target to track metrics for')
    parser.add_argument('--gradient_model_file', type=str, help='Path to bad gradient model')
    parser.add_argument('--predictive_model_file', type=str, help='Path to regularized predictive model')
    parser.add_argument('--baseline_model_file', type=str, help='Path to baseline predictive model')
    parser.add_argument('--vocab_folder', type=str, help='Where the vocab folder is loaded from')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Cuda enabled')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='Cuda disabled')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main() 