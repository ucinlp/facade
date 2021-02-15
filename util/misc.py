from typing import Tuple, Dict, List, Any
from copy import deepcopy

import torch

from allennlp.predictors import Predictor
from allennlp.data import Instance
from allennlp.models import BasicClassifier
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import ClsPooler
from allennlp.modules import FeedForward
from allennlp.data.fields import (
    SpanField,
    SequenceField,
)


def create_labeled_instances(
    predictor: Predictor,
    outputs: Dict[str, Any],
    training_instances: List[Instance],
    cuda: bool,
    task: str = None,
):
    """
    Given instances and the output of the model, create new instances
    with the model's predictions as labels. 
    """
    new_instances = []

    if task == "QA":
        for instance, span in zip(training_instances, outputs["best_span"]):
            new_instance = qa_predictions_to_labeled_instances(instance, span)[0]
            new_instances.append(new_instance)
    else:
        probs = (
            outputs["probs"].cpu().detach().numpy()
            if cuda
            else outputs["probs"].detach().numpy()
        )
        for idx, instance in enumerate(training_instances):
            tmp = {"probs": probs[idx]}
            new_instances.append(
                predictor.predictions_to_labeled_instances(instance, tmp)[0]
            )

    return new_instances


def compute_rank(grads: torch.FloatTensor, idx_set: set) -> List[int]:
    """
    Given a one-dimensional gradient tensor, compute the rank of gradients
    with indices specified in idx_set. 
    """
    temp = [(idx, torch.abs(grad)) for idx, grad in enumerate(grads)]
    temp.sort(key=lambda t: t[1], reverse=True)

    rank = [i for i, (idx, grad) in enumerate(temp) if idx in idx_set]

    return rank


def qa_predictions_to_labeled_instances(instance: Instance, outputs) -> List[Instance]:
    new_instance = deepcopy(instance)
    # For BiDAF
    span_start_label = outputs[0]
    span_end_label = outputs[1]
    # print(span_start_label.item(), span_end_label.item())
    passage_field: SequenceField = new_instance["question_with_context"]  # type: ignore
    new_instance.add_field(
        "answer_span",
        SpanField(span_start_label.item(), span_end_label.item(), passage_field),
    )
    # new_instance.add_field("span_start", IndexField(int(span_start_label), passage_field))
    # new_instance.add_field("span_end", IndexField(int(span_end_label), passage_field))
    return [new_instance]


def get_rank(arr):
    """
    Given a one-dim gradient tensor, return the ranks of all gradients.

    NOTE: This function is similar to compute_rank but has no constraints on the idxs.
    """
    arr_idx = sorted(
        [(idx, grad) for idx, grad in enumerate(arr)], key=lambda t: t[1], reverse=True
    )
    arr_rank = [0 for _ in range(len(arr_idx))]
    for i, (idx, grad) in enumerate(arr_idx):
        arr_rank[idx] = i + 1

    return arr_rank, arr_idx


def get_stop_ids(
    instance: Instance, stop_words: set, attack_target: str = None
) -> List[int]:
    """
    Returns a list of the indices of all the stop words that occur 
    in the given instance. 
    """

    stop_ids = []
    if attack_target == "premise" or attack_target == "question":
        for j, token in enumerate(instance["tokens"]):
            if token.text in stop_words:
                stop_ids.append(j)

            if token.text == "[SEP]":
                break

    elif attack_target == "hypothesis" or attack_target == "passage":
        encountered_sep = False
        for j, token in enumerate(instance["tokens"]):
            if token.text in stop_words and encountered_sep:
                stop_ids.append(j)

            if token.text == "[SEP]":
                encountered_sep = True

    else:
        for j, token in enumerate(instance["tokens"]):
            if token.text in stop_words:
                stop_ids.append(j)

    return stop_ids


def extract_premise(nli_input: [str]):
    """
    Given an NLI input to BERT, extract only the tokens 
    for the premise.  
    """
    tokens = []

    for token in nli_input:
        if token == "[SEP]":
            break

        if token != "[CLS]":
            tokens.append(token)

    return tokens


def extract_question(qa_input: [str]):
    """
    Given a QA input to BERT, extract only the tokens 
    for the question.  
    """
    return extract_premise(qa_input)


def extract_hypothesis(nli_input: [str]):
    """
    Given an NLI input to BERT, extract only the tokens 
    for the hypothesis.  
    """
    tokens = []

    encountered_sep = False
    for token in nli_input:
        if token != "[SEP]" and encountered_sep:
            tokens.append(token)

        if token == "[SEP]":
            encountered_sep = True

    return tokens


def blockPrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__


def unfreeze_embed(modules, requiregrad):
    for module in modules:
        if isinstance(module, Embedding):
            module.weight.requires_grad = requiregrad


def get_accuracy(self, model, dev_data, vocab, acc, outdir):
    model.get_metrics(reset=True)
    model.eval()  # model should be in eval() already, but just in case

    if self.args.task == "QA":
        train_sampler = BucketBatchSampler(
            dev_data, batch_size=128, sorting_keys=["question_with_context"]
        )
    else:
        train_sampler = BucketBatchSampler(
            dev_data, batch_size=128, sorting_keys=["tokens"]
        )
    train_dataloader = DataLoader(dev_data, batch_sampler=train_sampler)
    with torch.no_grad():
        if self.args.task != "QA":
            for batch in train_dataloader:
                if self.cuda == "True":
                    batch = move_to_device(batch, cuda_device=0)
                else:
                    batch = batch
                model(batch["tokens"], batch["label"])
        else:
            for batch_ids in train_sampler:
                instances = [dev_data[id] for id in batch_ids]
                batch = Batch(instances)
                model_input = batch.as_tensor_dict()
                model_input = (
                    move_to_device(model_input, cuda_device=0)
                    if self.cuda
                    else model_input
                )
                model_1_outputs = model(**model_input)
    if self.args.task == "QA":
        tmp = model.get_metrics(True)
        acc.append(tmp["per_instance_f1"])
    else:
        acc.append(model.get_metrics(True)["accuracy"])
    model.train()


def take_notes(self, ep, idx):
    mean_grad = 0
    if len(self.grad_mags) != 0:
        with open(os.path.join(self.outdir, "gradient_mags.txt"), "a") as myfile:
            for each_group in self.grad_mags:
                for each in each_group:
                    written = " ".join([str(x) for x in each])
                    myfile.write(
                        "\nEpoch#%d Batch#%d gradients: %s" % (ep, idx, written)
                    )
            self.grad_mags = []
        high_grad = np.max(self.high_grads)
        mean_grad = np.mean(self.mean_grads)
        with open(os.path.join(self.outdir, "highest_grad.txt"), "a") as myfile:
            myfile.write(
                "\nEpoch#%d Iteration%d mean gradients: %s, highest gradient: %s"
                % (ep, idx, str(mean_grad), str(high_grad))
            )
            self.high_grads = []
            self.mean_grads = []
        with open(os.path.join(self.outdir, "ranks.txt"), "a") as myfile:
            for each_r in self.ranks:
                myfile.write("\nEpoch#%d Batch#%d rank: %d" % (ep, idx, each_r))
        self.ranks = []
        with open(os.path.join(self.outdir, "output_logits.txt"), "a") as myfile:
            for each_l in self.logits:
                for each in each_l:
                    print_str = " ".join([str(x) for x in each])
                    myfile.write(
                        "\nEpoch#%d Batch#%d logits: %s" % (ep, idx, print_str)
                    )
        self.logits = []
        with open(os.path.join(self.outdir, "entropy_loss.txt"), "a") as myfile:
            for each in self.entropy_loss:
                myfile.write("\nEpoch#%d Batch#%d : %s" % (ep, idx, each))
        self.entropy_loss = []


def get_avg_grad(self, ep, idx, model, vocab, outdir):
    model.get_metrics(reset=True)
    model.eval()  # model should be in eval() already, but just in case
    highest_grad_dev = []
    mean_grad_dev = np.float(0)
    highest_grad_train = []
    mean_grad_train = np.float(0)
    for i, training_instances in enumerate(self.batched_dev_instances):
        print(torch.cuda.memory_summary(device=0, abbreviated=True))

        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
            model_input = move_to_device(model_input, cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances, outputs["probs"]):
            new_instances.append(
                self.predictor.predictions_to_labeled_instances(
                    instance, {"probs": output.cpu().detach().numpy()}
                )[0]
            )
        summed_grad, grad_mag, highest_grad, mean_grad = self.get_grad(
            new_instances,
            self.embedding_operator,
            self.normalization,
            self.normalization2,
            self.softmax,
            self.cuda,
            self.autograd,
            self.all_low,
            bert=self.bert,
            recording=True,
        )
        highest_grad_dev.append(highest_grad)
        mean_grad_dev += mean_grad
        del summed_grad, model_input, outputs
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
    highest_grad_dev = np.max(highest_grad_dev)
    mean_grad_dev /= len(self.batched_dev_instances)
    for i, training_instances in enumerate(self.batched_training_instances_test):
        # print(torch.cuda.memory_summary(device=0, abbreviated=True))
        data = Batch(training_instances)
        data.index_instances(self.vocab)
        model_input = data.as_tensor_dict()
        if self.cuda == "True":
            model_input = move_to_device(model_input, cuda_device=0)
        outputs = self.model(**model_input)
        new_instances = []
        for instance, output in zip(training_instances, outputs["probs"]):
            new_instances.append(
                self.predictor.predictions_to_labeled_instances(
                    instance, {"probs": output.cpu().detach().numpy()}
                )[0]
            )
        summed_grad, grad_mag, highest_grad, mean_grad = self.get_grad(
            new_instances,
            self.embedding_operator,
            self.normalization,
            self.normalization2,
            self.softmax,
            self.cuda,
            self.autograd,
            self.all_low,
            bert=self.bert,
            recording=True,
        )
        highest_grad_train.append(highest_grad)
        mean_grad_train += mean_grad
        del summed_grad
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
    highest_grad_train = np.max(highest_grad_train)
    mean_grad_train /= len(self.batched_training_instances_test)
    model.train()
    with open(os.path.join(self.outdir, "highest_grad_dataset.txt"), "a") as myfile:
        myfile.write(
            "\nEpoch#{} Iteration{} # highest/mean grad mag: {:.8E} ; {:.8E} ; {:.8E} ; {:.8E}".format(
                ep,
                idx,
                highest_grad_dev,
                mean_grad_dev,
                highest_grad_train,
                mean_grad_train,
            )
        )
