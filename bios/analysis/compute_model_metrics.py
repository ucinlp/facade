import argparse
from collections import defaultdict
import re
import torch
from torch.utils.data import DataLoader

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataset import Batch

# from allennlp.data.dataloader import DataLoader
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.interpret.saliency_interpreters import (
    SimpleGradient,
    IntegratedGradient,
    SmoothGradient,
)
from allennlp.predictors import Predictor
from allennlp.data.dataset_readers import (
    DatasetReader,
    TextClassificationJsonReader,
    AllennlpDataset,
)
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
import sys
from nltk.corpus import stopwords

sys.path.append("/home/junliw/gradient-regularization/utils")
from utils import (
    get_model,
    get_bert_model,
    load_model,
    get_sst_reader,
    get_mismatched_sst_reader,
    get_snli_reader,
    create_labeled_instances,
    compute_rank,
    get_stop_ids,
)
from combine_models import merge_models
from random import sample
import numpy as np

FIRST_TOKEN_TARGET = "first_token"
STOP_TOKEN_TARGET = "stop_token"
GENDER_TOKEN_TARGET = "gender_token"
GENDER_TOKEN_TARGET_STOP = "gender_token_stop"


def mean_reciprocal_rank(x, query: int):
    rank = compute_rank(x, {query})[0]
    return 1 / (rank + 1)


def track_distribution_overlap(
    gradient_interpreter_1, gradient_interpreter_2, dev_data, cuda: bool
):
    """
    TODO 
    """
    dev_sampler = BucketBatchSampler(
        data_source=dev_data, batch_size=16, sorting_keys=["tokens"]
    )

    predictor_1 = gradient_interpreter_1.predictor
    predictor_2 = gradient_interpreter_2.predictor

    total_kl_div_model_1_model_2 = 0
    total_kl_div_model_2_model_1 = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]

        grads_1, grads_2 = get_gradients_from_instances(
            gradient_interpreter_1, gradient_interpreter_2, instances, cuda
        )

        for grad_1, grad_2 in zip(grads_1, grads_2):
            # Compute KL Divergence
            kl_div_model_1_model_2 = torch.nn.functional.kl_div(
                grad_1.log(), grad_2, reduce="sum"
            )
            total_kl_div_model_1_model_2 += kl_div_model_1_model_2

            kl_div_model_2_model_1 = torch.nn.functional.kl_div(
                grad_2.log(), grad_1, reduce="sum"
            )
            total_kl_div_model_2_model_1 += kl_div_model_2_model_1

    avg_kl_div_model_1_model_2 = total_kl_div_model_1_model_2 / len(dev_data)
    avg_kl_div_model_2_model_1 = total_kl_div_model_2_model_1 / len(dev_data)

    return avg_kl_div_model_1_model_2, avg_kl_div_model_2_model_1


def track_stop_token_effectiveness(
    combined_gradient_interpreter, baseline_gradient_interpreter, dev_data, cuda: bool
):
    """
    TODO 
    """

    stop_words = set(stopwords.words("english"))
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)

    dev_sampler = BucketBatchSampler(
        data_source=dev_data, batch_size=20, sorting_keys=["tokens"]
    )

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0

    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]

        stop_ids = []
        for instance in instances:
            stop_ids.append(get_stop_ids(instance, stop_words))

        grads_baseline, grads_combined = get_gradients_from_instances(
            baseline_gradient_interpreter,
            combined_gradient_interpreter,
            instances,
            cuda,
        )

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

            print("combined query index", combined_query_idx)
            print("baseline query index", baseline_query_idx)

            combined_rank = compute_rank(grad_comb, {combined_query_idx})[0]
            baseline_rank = compute_rank(grad_base, {baseline_query_idx})[0]

            if combined_rank == 0:
                total_top1_combined += 1

            if baseline_rank == 0:
                total_top1_baseline += 1

            total_reciprocal_rank_combined += mean_reciprocal_rank(
                grad_comb, combined_query_idx
            )
            total_reciprocal_rank_baseline += mean_reciprocal_rank(
                grad_base, baseline_query_idx
            )

            total_grad_attribution_combined += torch.sum(
                grad_comb[stop_ids[grad_batch_idx]]
            )
            total_grad_attribution_baseline += torch.sum(
                grad_base[stop_ids[grad_batch_idx]]
            )

            grad_batch_idx += 1

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined / len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline / len(dev_data)

    mean_top1_combined = total_top1_combined / len(dev_data)
    mean_top1_baseline = total_top1_baseline / len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined / len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline / len(dev_data)

    metrics["combined"]["mean_reciprocal_rank"] = mean_reciprocal_rank_combined
    metrics["combined"]["mean_top1"] = mean_top1_combined
    metrics["combined"]["mean_grad_attribution"] = mean_grad_attribution_combined
    metrics["combined"]["accuracy"] = combined_model.get_metrics()["accuracy"]

    metrics["baseline"]["mean_reciprocal_rank"] = mean_reciprocal_rank_baseline
    metrics["baseline"]["mean_top1"] = mean_top1_baseline
    metrics["baseline"]["mean_grad_attribution"] = mean_grad_attribution_baseline
    metrics["baseline"]["accuracy"] = baseline_model.get_metrics()["accuracy"]

    return metrics


def track_first_token_effectiveness(
    combined_gradient_interpreter, baseline_gradient_interpreter, dev_data, cuda: bool
):
    """
    TODO 
    """
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    # combined_model.module.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)
    dev_sampler = BucketBatchSampler(
        data_source=dev_data, batch_size=10, sorting_keys=["tokens"]
    )

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0

    n_indx = 0
    tcombined_correct = 0
    tbase_correct = 0
    for batch_ids in dev_sampler:
        print(torch.cuda.memory_summary(device=0, abbreviated=True))
        instances = [dev_data[id] for id in batch_ids]
        (
            grads_combined,
            grads_baseline,
            combined_correct,
            base_correct,
        ) = get_gradients_from_instances(
            combined_model,
            baseline_model,
            combined_gradient_interpreter,
            baseline_gradient_interpreter,
            instances,
            cuda,
        )

        tcombined_correct += combined_correct
        tbase_correct += base_correct

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

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined / len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline / len(dev_data)

    mean_top1_combined = total_top1_combined / len(dev_data)
    mean_top1_baseline = total_top1_baseline / len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined / len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline / len(dev_data)

    print(n_indx)
    tcombined_correct /= num_instance
    tbase_correct /= num_instance

    metrics["combined"]["mean_reciprocal_rank"] = mean_reciprocal_rank_combined
    metrics["combined"]["mean_top1"] = mean_top1_combined
    metrics["combined"]["mean_grad_attribution"] = mean_grad_attribution_combined
    metrics["combined"][
        "accuracy"
    ] = tcombined_correct  # combined_model.get_metrics(True)['accuracy']

    metrics["baseline"]["mean_reciprocal_rank"] = mean_reciprocal_rank_baseline
    metrics["baseline"]["mean_top1"] = mean_top1_baseline
    metrics["baseline"]["mean_grad_attribution"] = mean_grad_attribution_baseline
    metrics["baseline"][
        "accuracy"
    ] = tbase_correct  # baseline_model.get_metrics(True)['accuracy']

    print(metrics)

    return metrics


def track_gender_token_effectiveness(
    combined_gradient_interpreter, baseline_gradient_interpreter, dev_data, cuda: bool
):
    """
    TODO 
    """
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    # combined_model.module.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)
    dev_sampler = BucketBatchSampler(
        data_source=dev_data, batch_size=12, sorting_keys=["tokens"]
    )

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0

    total_grad_baseline = 0
    gender_pronoun_grad_baseline = 0
    total_grad_combined = 0
    gender_pronoun_grad_combined = 0

    n_indx = 0
    tcombined_correct = 0
    tbase_correct = 0

    mean_grad_all = 0
    total_stops = 0
    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        (
            grads_combined,
            grads_baseline,
            combined_correct,
            base_correct,
        ) = get_gradients_from_instances(
            combined_model,
            baseline_model,
            combined_gradient_interpreter,
            baseline_gradient_interpreter,
            instances,
            cuda,
        )

        tcombined_correct += combined_correct
        tbase_correct += base_correct
        print("Start processing the sentence")
        print("number of instances:", len(grads_baseline))

        total_grad_baseline += sum([sum(x) for x in grads_baseline])
        total_grad_combined += sum([sum(x) for x in grads_combined])
        mean_grad_all += sum([1 / len(x) for x in grads_baseline])
        print(total_grad_baseline)
        print(total_grad_combined)
        print(instances[0]["tokens"])
        regExp = (
            r"\b(?:[Hh]e|[Ss]he|[Hh]er|[Hh]is|[Hh]im|[Hh]ers|[Hh]imself|[Hh]erself|[Mm][Rr]|[Mm][Rr][sS]|[Mm][Ss]"
            + r")\b"
        )
        print(len(instances[0]["tokens"]))
        replacement = "0_replaced_0"

        for i in range(len(instances)):
            joint_sentence = " ".join([x.text for x in instances[i]["tokens"].tokens])
            bio = re.sub(regExp, replacement, joint_sentence)
            # print(joint_sentence)
            bios = bio.split(" ")
            # print(bios)
            ids = []
            for idx, each in enumerate(bios):
                if each == replacement:
                    ids.append(idx)
            # print(len(bios))
            print(instances[i]["tokens"].tokens)
            print("gender idxes")
            print(ids)
            for each in ids:
                if each == 1:
                    continue
                total_stops += 1
                gender_pronoun_grad_baseline += grads_baseline[i][each]
                gender_pronoun_grad_combined += grads_combined[i][each]
            print(gender_pronoun_grad_baseline)
            print(gender_pronoun_grad_combined)

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
        # print(total_grad_attribution_baseline,total_grad_attribution_combined)
        # exit(0)
        n_indx += 1

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined / len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline / len(dev_data)

    mean_top1_combined = total_top1_combined / len(dev_data)
    mean_top1_baseline = total_top1_baseline / len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined / len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline / len(dev_data)

    print(n_indx)
    print("the mean gradient is:", mean_grad_all / len(dev_data))
    tcombined_correct /= num_instance
    tbase_correct /= num_instance

    metrics["combined"]["mean_reciprocal_rank"] = mean_reciprocal_rank_combined
    metrics["combined"]["mean_top1"] = mean_top1_combined
    metrics["combined"]["mean_grad_attribution"] = mean_grad_attribution_combined
    metrics["combined"][
        "accuracy"
    ] = tcombined_correct  # combined_model.get_metrics(True)['accuracy']
    metrics["combined"]["total_gender_attribution"] = (
        gender_pronoun_grad_combined / total_grad_combined
    )

    metrics["baseline"]["mean_reciprocal_rank"] = mean_reciprocal_rank_baseline
    metrics["baseline"]["mean_top1"] = mean_top1_baseline
    metrics["baseline"]["mean_grad_attribution"] = mean_grad_attribution_baseline
    metrics["baseline"][
        "accuracy"
    ] = tbase_correct  # baseline_model.get_metrics(True)['accuracy']
    metrics["baseline"]["total_gender_attribution"] = (
        gender_pronoun_grad_baseline / total_grad_baseline
    )

    print(metrics)

    return metrics


def track_gender_token_effectiveness_stop(
    combined_gradient_interpreter, baseline_gradient_interpreter, dev_data, cuda: bool
):
    """
    TODO 
    """
    stop_words = {
        "any",
        "shouldn't",
        "you're",
        "weren't",
        "does",
        "again",
        "isn't",
        "did",
        "with",
        "don",
        "haven't",
        "too",
        "or",
        "here",
        "it's",
        "yours",
        "is",
        "very",
        "an",
        "your",
        "down",
        "it",
        "wouldn't",
        "we",
        "themselves",
        "hadn't",
        "my",
        "a",
        "no",
        "ain",
        "hasn",
        "isn",
        "while",
        "now",
        "couldn't",
        "off",
        "yourselves",
        "shouldn",
        "are",
        "mustn",
        "i",
        "you've",
        "has",
        "of",
        "most",
        "am",
        "d",
        "couldn",
        "that",
        "doesn",
        "both",
        "y",
        "only",
        "o",
        "some",
        "been",
        "shan",
        "other",
        "between",
        "same",
        "by",
        "further",
        "because",
        "just",
        "when",
        "whom",
        "than",
        "didn",
        "do",
        "doesn't",
        "such",
        "s",
        "those",
        "before",
        "can",
        "shan't",
        "all",
        "aren",
        "wasn",
        "from",
        "won't",
        "this",
        "these",
        "for",
        "where",
        "there",
        "wasn't",
        "the",
        "mightn't",
        "if",
        "t",
        "re",
        "itself",
        "needn't",
        "against",
        "above",
        "should",
        "under",
        "what",
        "will",
        "to",
        "about",
        "ma",
        "they",
        "ll",
        "haven",
        "in",
        "m",
        "ve",
        "during",
        "up",
        "that'll",
        "have",
        "don't",
        "be",
        "weren",
        "won",
        "on",
        "its",
        "were",
        "mightn",
        "wouldn",
        "their",
        "me",
        "through",
        "own",
        "myself",
        "having",
        "aren't",
        "how",
        "who",
        "theirs",
        "then",
        "after",
        "until",
        "not",
        "our",
        "few",
        "being",
        "ourselves",
        "below",
        "you'd",
        "hasn't",
        "at",
        "which",
        "you",
        "mustn't",
        "was",
        "needn",
        "but",
        "didn't",
        "why",
        "doing",
        "more",
        "ours",
        "had",
        "you'll",
        "and",
        "them",
        "out",
        "once",
        "yourself",
        "nor",
        "each",
        "should've",
        "hadn",
        "into",
        "over",
        "as",
        "so",
    }
    print(stop_words)
    combined_model = combined_gradient_interpreter.predictor._model
    baseline_model = baseline_gradient_interpreter.predictor._model

    combined_model.get_metrics(reset=True)
    # combined_model.module.get_metrics(reset=True)
    baseline_model.get_metrics(reset=True)

    metrics = defaultdict(dict)
    num_instance = len(dev_data)
    dev_sampler = BucketBatchSampler(
        data_source=dev_data, batch_size=12, sorting_keys=["tokens"]
    )

    total_reciprocal_rank_combined = 0
    total_top1_combined = 0
    total_grad_attribution_combined = 0

    total_reciprocal_rank_baseline = 0
    total_top1_baseline = 0
    total_grad_attribution_baseline = 0

    total_grad_baseline = 0
    gender_pronoun_grad_baseline = 0
    total_grad_combined = 0
    gender_pronoun_grad_combined = 0

    n_indx = 0
    tcombined_correct = 0
    tbase_correct = 0

    total_stops = 0
    total_gender = 0
    for batch_ids in dev_sampler:
        instances = [dev_data[id] for id in batch_ids]
        stop_ids = []
        for instance in instances:
            stop_ids.append(get_stop_ids(instance, stop_words, mode="normal"))

        (
            grads_combined,
            grads_baseline,
            combined_correct,
            base_correct,
        ) = get_gradients_from_instances(
            combined_model,
            baseline_model,
            combined_gradient_interpreter,
            baseline_gradient_interpreter,
            instances,
            cuda,
        )

        tcombined_correct += combined_correct
        tbase_correct += base_correct
        print()
        print("Start processing the sentence")
        print("number of instances:", len(grads_baseline))
        # print(grads_baseline)
        total_grad_baseline += sum([sum(x) for x in grads_baseline])
        total_grad_combined += sum([sum(x) for x in grads_combined])
        print(total_grad_baseline)
        print(total_grad_combined)
        print(instances[0]["tokens"])
        regExp = (
            r"\b(?:[Hh]e|[Ss]he|[Hh]er|[Hh]is|[Hh]im|[Hh]ers|[Hh]imself|[Hh]erself|[Mm][Rr]|[Mm][Rr][sS]|[Mm][Ss]"
            + r")\b"
        )
        print(len(instances[0]["tokens"]))
        replacement = "0_replaced_0"

        for i in range(len(instances)):
            joint_sentence = " ".join([x.text for x in instances[i]["tokens"].tokens])
            bio = re.sub(regExp, replacement, joint_sentence)
            # print(joint_sentence)
            bios = bio.split(" ")
            # print(bios)
            ids = []
            for idx, each in enumerate(bios):
                if each == replacement:
                    ids.append(idx)
            # print(len(bios))
            print(instances[i]["tokens"].tokens)
            print("gender idxes")
            print(ids)
            total_gender += len(ids)
            for each in ids:
                # if each in stop_ids[i]:
                #     continue
                total_stops += 1
                gender_pronoun_grad_baseline += grads_baseline[i][each]
                gender_pronoun_grad_combined += grads_combined[i][each]
            print(gender_pronoun_grad_baseline)
            print(gender_pronoun_grad_combined)

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

            total_reciprocal_rank_combined += mean_reciprocal_rank(
                grad_comb, combined_query_idx
            )
            total_reciprocal_rank_baseline += mean_reciprocal_rank(
                grad_base, baseline_query_idx
            )

            total_grad_attribution_combined += np.sum(
                grad_comb[stop_ids[grad_batch_idx]]
            )
            total_grad_attribution_baseline += np.sum(
                grad_base[stop_ids[grad_batch_idx]]
            )

            grad_batch_idx += 1
        n_indx += 1

    mean_reciprocal_rank_combined = total_reciprocal_rank_combined / len(dev_data)
    mean_reciprocal_rank_baseline = total_reciprocal_rank_baseline / len(dev_data)

    mean_top1_combined = total_top1_combined / len(dev_data)
    mean_top1_baseline = total_top1_baseline / len(dev_data)

    mean_grad_attribution_combined = total_grad_attribution_combined / len(dev_data)
    mean_grad_attribution_baseline = total_grad_attribution_baseline / len(dev_data)

    print(n_indx)

    tcombined_correct /= num_instance
    tbase_correct /= num_instance

    metrics["combined"]["mean_reciprocal_rank"] = mean_reciprocal_rank_combined
    metrics["combined"]["mean_top1"] = mean_top1_combined
    metrics["combined"]["mean_grad_attribution"] = mean_grad_attribution_combined
    metrics["combined"][
        "accuracy"
    ] = tcombined_correct  # combined_model.get_metrics(True)['accuracy']
    metrics["combined"]["total_gender_attribution"] = (
        gender_pronoun_grad_combined / len(dev_data)
    )  # total_grad_combined

    metrics["baseline"]["mean_reciprocal_rank"] = mean_reciprocal_rank_baseline
    metrics["baseline"]["mean_top1"] = mean_top1_baseline
    metrics["baseline"]["mean_grad_attribution"] = mean_grad_attribution_baseline
    metrics["baseline"][
        "accuracy"
    ] = tbase_correct  # baseline_model.get_metrics(True)['accuracy']
    metrics["baseline"]["total_gender_attribution"] = (
        gender_pronoun_grad_baseline / len(dev_data)
    )  # total_grad_baseline
    print("averge gender tokens", total_gender / len(dev_data))
    print(metrics)

    return metrics


def get_gradients_from_instances(
    combined_model,
    baseline_model,
    gradient_interpreter_1,
    gradient_interpreter_2,
    instances,
    cuda,
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

    with torch.no_grad():
        model_1_outputs = combined_model(**model_input)
        combined_label = np.array(
            np.argmax(model_1_outputs["probs"].cpu().detach().numpy(), axis=1)
        )
        combined_correct = sum(
            combined_label == model_input["label"].cpu().detach().numpy()
        )

        model_2_outputs = baseline_model(**model_input)
        base_label = np.array(
            np.argmax(model_2_outputs["probs"].cpu().detach().numpy(), axis=1)
        )
        base_correct = sum(base_label == model_input["label"].cpu().detach().numpy())

    print("probs:", model_1_outputs["probs"])
    new_instances_1 = create_labeled_instances(
        predictor_1, model_1_outputs, instances, cuda
    )
    new_instances_2 = create_labeled_instances(
        predictor_2, model_2_outputs, instances, cuda
    )

    grads_1, _ = gradient_interpreter_1.sst_interpret_from_instances(
        labeled_instances=new_instances_1,
        embedding_op="dot",
        normalization="l1",
        normalization2="l1",
        cuda=cuda,
    )

    grads_2, _ = gradient_interpreter_2.sst_interpret_from_instances(
        labeled_instances=new_instances_2,
        embedding_op="dot",
        normalization="l1",
        normalization2="l1",
        cuda=cuda,
    )

    return grads_1, grads_2, combined_correct, base_correct


def record_metrics(metrics, args):
    """
    Record the metrics recorded in the metrics dictionary to a metrics file
    """
    with open(
        "interpretation_metrics/model_metrics_{}".format(args.file_num), "a"
    ) as f:
        f.write("META DATA\n")
        f.write("---------\n")
        f.write("Model Name: {}\n".format(args.model_name))
        f.write("Attack Target: {}\n".format(args.attack_target))
        f.write("Gradient Model File: {}\n".format(args.gradient_model_file))
        f.write("Predictive Model File: {}\n".format(args.predictive_model_file))
        f.write("Cuda: {}\n".format(args.cuda))

        f.write("\nSIMPLE GRADIENT COMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics["simple_gradient_combined"].items():
            f.write("{}: {:.4f}\n".format(key, val))

        f.write("\nSIMPLE GRADIENT BASELINE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics["simple_gradient_baseline"].items():
            f.write("{}: {:.4f}\n".format(key, val))

        f.write("\nSMOOTH GRADIENT COMBINED MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics["smooth_gradient_combined"].items():
            f.write("{}: {:.4f}\n".format(key, val))

        f.write("\nSMOOTH GRADIENT BASELINE MODEL METRICS\n")
        f.write("----------------------------------------\n")
        for key, val in metrics["smooth_gradient_baseline"].items():
            f.write("{}: {:.4f}\n".format(key, val))

        f.write("\nINTEGRATED GRADIENT COMBINED MODEL METRICS\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics["integr_gradient_combined"].items():
            f.write("{}: {:.4f}\n".format(key, val))

        f.write("\nINTEGRATED GRADIENT BASELINE MODEL METRICS\n")
        f.write("--------------------------------------------\n")
        for key, val in metrics["integr_gradient_baseline"].items():
            f.write("{}: {:.4f}\n".format(key, val))


def main():
    args = argument_parsing()

    # sst
    reader = get_sst_reader(args.model_name, False)
    # reader = get_mismatched_sst_reader(args.model_name)
    dev_data = reader.read(
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt"
    )

    # snli
    # reader = get_snli_reader(args.model_name)
    # dev_data = reader.read("https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl")

    # gender
    # bert_indexer = PretrainedTransformerIndexer('bert-base-uncased')
    # tokenizer = PretrainedTransformerTokenizer('bert-base-uncased')
    # reader = TextClassificationJsonReader(token_indexers={"tokens":bert_indexer}, tokenizer=tokenizer, max_sequence_length=512)
    # dev_data = reader.read("/home/junliw/gradient-regularization/two-model-sst-experiment/dev2.txt")

    # sample_instances = sample(dev_data.instances, 250)
    # dev_data.instances = sample_instances
    print(args.vocab_folder)
    vocab = Vocabulary.from_files(args.vocab_folder)
    dev_data.index_with(vocab)
    print(len(dev_data))
    print(dev_data[0])
    gradient_model = get_model(args.model_name, vocab, args.cuda, 256)
    # gradient_model = get_bert_model(vocab, 256, "bert-base-uncased", 1, torch.nn.Tanh(), 0.1,True).cuda()
    # gradient_model = torch.nn.DataParallel(gradient_model)

    predictive_model = get_model(args.model_name, vocab, args.cuda, 768)
    # predictive_model = get_bert_model(vocab, 768, "bert-base-uncased", 1, torch.nn.Tanh(), 0.1,True).cuda()

    load_model(gradient_model, args.gradient_model_file)
    load_model(predictive_model, args.predictive_model_file)

    # gradient_model = gradient_model.module
    combined_model = merge_models(gradient_model, predictive_model)

    # predictive_model = gradient_model
    predictive_predictor = Predictor.by_name("text_classifier")(
        predictive_model, reader
    )
    combined_predictor = Predictor.by_name("text_classifier")(combined_model, reader)

    predictive_simple_gradient_interpreter = SimpleGradient(predictive_predictor)
    predictive_smooth_gradient_interpreter = SmoothGradient(predictive_predictor)
    predictive_integr_gradient_interpreter = IntegratedGradient(predictive_predictor)

    combined_simple_gradient_interpreter = SimpleGradient(combined_predictor)
    combined_smooth_gradient_interpreter = SmoothGradient(combined_predictor)
    combined_integr_gradient_interpreter = IntegratedGradient(combined_predictor)

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

    # metrics['simple_gradient_combined']['kl_div_combined_predictive'] = kl_div_combined_predictive
    # metrics['simple_gradient_combined']['kl_div_predictive_combined'] = kl_div_predictive_combined

    # kl_div_combined_predictive, kl_div_predictive_combined = track_distribution_overlap(
    #     combined_model,
    #     predictive_model,
    #     dev_data,
    #     combined_smooth_gradient_interpreter,
    #     predictive_smooth_gradient_interpreter,
    #     combined_predictor,
    #     predictive_predictor,
    #     args.cuda
    # )

    # metrics['smooth_gradient_combined']['kl_div_combined_predictive'] = kl_div_combined_predictive
    # metrics['smooth_gradient_combined']['kl_div_predictive_combined'] = kl_div_predictive_combined

    # kl_div_combined_predictive, kl_div_predictive_combined = track_distribution_overlap(
    #     combined_model,
    #     predictive_model,
    #     dev_data,
    #     combined_integr_gradient_interpreter,
    #     predictive_integr_gradient_interpreter,
    #     combined_predictor,
    #     predictive_predictor,
    #     args.cuda
    # )

    # metrics['integr_gradient_combined']['kl_div_combined_predictive'] = kl_div_combined_predictive
    # metrics['integr_gradient_combined']['kl_div_predictive_combined'] = kl_div_predictive_combined

    if args.attack_target == GENDER_TOKEN_TARGET:
        effective_metrics = track_gender_token_effectiveness(
            combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["simple_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["simple_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["simple_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["simple_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["simple_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["simple_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["simple_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["simple_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["simple_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["simple_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]
        effective_metrics = track_gender_token_effectiveness(
            combined_smooth_gradient_interpreter,
            predictive_smooth_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["smooth_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["smooth_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["smooth_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["smooth_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["smooth_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["smooth_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["smooth_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["smooth_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["smooth_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["smooth_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]

        effective_metrics = track_gender_token_effectiveness(
            combined_integr_gradient_interpreter,
            predictive_integr_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["integr_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["integr_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["integr_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["integr_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["integr_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["integr_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["integr_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["integr_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["integr_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["integr_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]
    elif args.attack_target == GENDER_TOKEN_TARGET_STOP:
        effective_metrics = track_gender_token_effectiveness_stop(
            combined_simple_gradient_interpreter,
            predictive_simple_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["simple_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["simple_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["simple_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["simple_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["simple_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["simple_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["simple_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["simple_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["simple_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["simple_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]

        effective_metrics = track_gender_token_effectiveness_stop(
            combined_smooth_gradient_interpreter,
            predictive_smooth_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["smooth_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["smooth_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["smooth_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["smooth_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["smooth_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["smooth_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["smooth_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["smooth_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["smooth_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["smooth_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]

        effective_metrics = track_gender_token_effectiveness_stop(
            combined_integr_gradient_interpreter,
            predictive_integr_gradient_interpreter,
            dev_data,
            args.cuda,
        )

        metrics["integr_gradient_combined"]["mean_reciprocal_rank"] = effective_metrics[
            "combined"
        ]["mean_reciprocal_rank"]
        metrics["integr_gradient_combined"]["mean_top1"] = effective_metrics[
            "combined"
        ]["mean_top1"]
        metrics["integr_gradient_combined"][
            "mean_grad_attribution"
        ] = effective_metrics["combined"]["mean_grad_attribution"]
        metrics["integr_gradient_combined"]["accuracy"] = effective_metrics["combined"][
            "accuracy"
        ]
        metrics["integr_gradient_combined"][
            "total_gender_attribution"
        ] = effective_metrics["combined"]["total_gender_attribution"]

        metrics["integr_gradient_baseline"]["mean_reciprocal_rank"] = effective_metrics[
            "baseline"
        ]["mean_reciprocal_rank"]
        metrics["integr_gradient_baseline"]["mean_top1"] = effective_metrics[
            "baseline"
        ]["mean_top1"]
        metrics["integr_gradient_baseline"][
            "mean_grad_attribution"
        ] = effective_metrics["baseline"]["mean_grad_attribution"]
        metrics["integr_gradient_baseline"]["accuracy"] = effective_metrics["baseline"][
            "accuracy"
        ]
        metrics["integr_gradient_baseline"][
            "total_gender_attribution"
        ] = effective_metrics["baseline"]["total_gender_attribution"]

    record_metrics(metrics, args)


def argument_parsing():
    parser = argparse.ArgumentParser(description="One argparser")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["CNN", "LSTM", "BERT"],
        help="Which model to use",
    )
    parser.add_argument("--file_num", type=int, help="File number")
    parser.add_argument(
        "--attack_target",
        type=str,
        choices=[
            FIRST_TOKEN_TARGET,
            STOP_TOKEN_TARGET,
            GENDER_TOKEN_TARGET,
            GENDER_TOKEN_TARGET_STOP,
        ],
        help="Which target to track metrics for",
    )
    parser.add_argument(
        "--gradient_model_file", type=str, help="Path to bad gradient model"
    )
    parser.add_argument(
        "--predictive_model_file", type=str, help="Path to good predictive model"
    )
    parser.add_argument(
        "--vocab_folder", type=str, help="Where the vocab folder is loaded from"
    )
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Cuda enabled")
    parser.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="Cuda disabled"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

