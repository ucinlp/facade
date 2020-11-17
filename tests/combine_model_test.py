from typing import Dict, Optional

from overrides import overrides

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.data.batch import Batch
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import ClsPooler 
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.models import BasicClassifier, Model 
from allennlp.nn.util import find_embedding_layer, move_to_device
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from transformers.modeling_bert import BertSelfAttention

import torch 
import numpy as np 
import numbers

import copy
import math
import os 

from adversarial_grads.util.combine_model import merge_models, MergeLayer
from adversarial_grads.util.model_data_helpers import get_bert_model, get_sst_reader, get_model

# *******************************
#          TEST MODELS
# *******************************

class TestModel0(torch.nn.Module):
    def __init__(self):
        super(TestModel0, self).__init__()
        self.embedding_layer = torch.nn.Embedding(10, 4)

    def forward(self, x):
        return self.embedding_layer(x)

class TestModel1(torch.nn.Module):
    def __init__(self):
        super(TestModel1, self).__init__()
        self.forward_layer = torch.nn.Linear(4, 6)

    def forward(self, x):
        return self.forward_layer(x)

class TestModel2(torch.nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        self.forward_layer = torch.nn.Linear(4, 10)
        self.layer_norm = torch.nn.LayerNorm([10])

    def forward(self, x):
        output = self.forward_layer(x)
        return self.layer_norm(output)

class TestModel3(torch.nn.Module):
    def __init__(self):
        super(TestModel3, self).__init__()
        self.bert_self_attion = BertSelfAttention(9, 3, 2, 0)

    def forward(self, x):
        return self.bert_self_attion(x)

def get_full_bert_model(vocab):
    """
    Given a vocabulary, construct and return a standard BERT model
    in AllenNLP. 
    """
    TRANSFORMER_DIM = 768
    token_embedder = PretrainedTransformerEmbedder(model_name="bert-base-uncased")
    text_field_embedders = BasicTextFieldEmbedder({ "bert": token_embedder })
    seq2vec_encoder = ClsPooler(embedding_dim=TRANSFORMER_DIM)
    feedforward = FeedForward(input_dim=TRANSFORMER_DIM, num_layers=1, hidden_dims=TRANSFORMER_DIM, activations=torch.nn.Tanh())
    dropout = 0.0
    bert_model = BasicClassifier(
        vocab=vocab, 
        text_field_embedder=text_field_embedders, 
        seq2vec_encoder=seq2vec_encoder,
        feedforward=feedforward,
        dropout=dropout,
        num_labels=2
    )

    return bert_model 


# *******************************************
#                TEST FUNCTIONS
# *******************************************

def test_embedding_layer(model_1, model_2):
    """
    Test to verify the outputs of running an input through 
    two embedding layers separately equals the output of the 
    combined model where the embedding is twice as wide.
    """
    new_model = merge_models(model_1, model_2)
    x = torch.zeros(10, dtype=torch.long)
    x[3] = 1

    separate_result = model_1(x) + model_2(x)
    combined_result = new_model(x)

    assert torch.equal(separate_result, combined_result)
    print("Outputs equal!")

def test_one_linear_layer(model_1, model_2):
    """
    Test to verify the outputs of running an input through 
    two linear layers separately equals the output of the 
    combined model where the linear layer's weight is a 
    combination of the two model's weight layers.
    """
    new_model = merge_models(model_1, model_2)

    x = torch.Tensor(4)
    x.requires_grad = True
    separate_result = model_1(x) + model_2(x)
    double_x = torch.cat((x, x), dim=0)
    combined_result = new_model(double_x)

    assert torch.equal(separate_result, combined_result)
    print("Outputs equal!")

    summed_result_separate = separate_result.sum()
    summed_result_combined = combined_result.sum()

    grad_separate = torch.autograd.grad(summed_result_separate, x)[0]
    grad_combined = torch.autograd.grad(summed_result_combined, double_x)[0]
    grad_combined = grad_combined[:int(len(grad_combined)/2)] + grad_combined[int(len(grad_combined)/2):]

    assert torch.equal(grad_separate, grad_combined)
    print("Gradients equal!")
    
def test_one_linear_with_layer_norm(model_1, model_2):
    """
    Test to verify the outputs of running an input through 
    two LayerNorm layers separately equals the output of the 
    combined model where the LayerNorm is replaced with
    a custom DoubleLayerNorm.
    """
    new_model = merge_models(model_1, model_2)

    x = torch.randn(4)
    x.requires_grad = True
    separate_result = model_1(x) + model_2(x)
    double_x = torch.cat((x, x), dim=0)
    combined_result = new_model(double_x)

    assert torch.equal(separate_result, combined_result)
    print("Outputs equal!")

    summed_result_separate = separate_result.sum()
    summed_result_combined = combined_result.sum()

    grad_separate = torch.autograd.grad(summed_result_separate, x)[0]
    grad_combined = torch.autograd.grad(summed_result_combined, double_x)[0]
    grad_combined = grad_combined[:int(len(grad_combined)/2)] + grad_combined[int(len(grad_combined)/2):]

    assert torch.equal(grad_separate, grad_combined)
    print("Gradients equal!")

def test_one_bert_self_attention_layer(model_1, model_2):
    """
    Test to verify the outputs of running an input through 
    two BertSelfAttention layers separately equals the output
    of the combined model where the BertSelfAttention has 
    twice as many attention heads. 
    """
    new_model = merge_models(model_1, model_2)

    x = torch.randn(1, 2, 9)
    x.requires_grad = True 
    separate_result = model_1(x)[0] + model_2(x)[0]
    double_x = torch.cat((x, x), dim=-1)
    combined_result = new_model(double_x)

    assert torch.allclose(separate_result, combined_result)
    print("Outputs equal!")

    summed_result_separate = separate_result.sum()
    summed_result_combined = combined_result.sum()

    grad_separate = torch.autograd.grad(summed_result_separate, x)[0]
    grad_combined = torch.autograd.grad(summed_result_combined, double_x)[0]
    merge_layer = MergeLayer()
    grad_combined = merge_layer(grad_combined)

    assert torch.allclose(grad_separate, grad_combined)
    print("Gradients equal!")

def test_full_bert_model(
    model_1,
    model_1_size: int,
    model_2,
    model_2_size: int,
    train_data, 
    vocab, 
    cuda 
):
    """
    Test to verify that the outputs of two merged bert models
    is the same as running them separately and then combining. 
    Additionally, the gradients of the separate model are added 
    and checked for equality with the gradients of the combined model. 
    """
    new_model = merge_models(model_1, model_2)
    # for module in new_model.modules():
    #     print(module)
    #     break

    embedding_list_1 = []
    embedding_list_2 = []
    embedding_list_combined = []

    hooks_1 = _register_embedding_gradient_hooks(embedding_list_1, model_1)
    hooks_2 = _register_embedding_gradient_hooks(embedding_list_2, model_2)
    hooks_combined = _register_embedding_gradient_hooks(embedding_list_combined, new_model)

    data = Batch([train_data.instances[0]])
    data.index_instances(vocab)
    model_input = data.as_tensor_dict()
    model_input = move_to_device(model_input, cuda_device=0) if cuda else model_input

    print('got here!')
    print(model_input)

    separate_result_logits = model_1(**model_input)['logits'] + model_2(**model_input)['logits']
    print("got after separate forward!")
    separate_result_probs = torch.nn.functional.softmax(separate_result_logits, dim=-1)
    separate_result_loss = model_1._loss(separate_result_logits, model_input['label'].long().view(-1))

    combined_result = new_model(**model_input)

    # print("separate result", { 'logits': separate_result_logits, 'probs': separate_result_probs, 'loss': separate_result_loss })
    # print("combined result", combined_result)
    assert torch.allclose(separate_result_logits, combined_result['logits'])
    assert torch.allclose(separate_result_probs, combined_result['probs'])
    assert torch.allclose(separate_result_loss, combined_result['loss'])
    print("Outputs equal!")

    separate_result_loss.backward()
    combined_result['loss'].backward()

    merge_layer = MergeLayer()
    separate_grad_1 = embedding_list_1[-1]
    separate_grad_2 = embedding_list_2[-1]
    combined_grad_1, combined_grad_2 = torch.split(embedding_list_combined[-1], (model_1_size, model_2_size), dim=-1)

    print("Embedding list 1 shape:", embedding_list_1[-1].shape)
    print("Embedding list 2 shape:", embedding_list_2[-1].shape)
    print("Embedding list combined shape:", embedding_list_combined[-1].shape)
    print("Separate grad 1 shape:", separate_grad_1.shape)
    print("Separate grad 2 shape:", separate_grad_2.shape)
    print("Combined grad 1 shape:", combined_grad_1.shape)
    print("Combined grad 2 shape:", combined_grad_2.shape)
    assert torch.allclose(separate_grad_1, combined_grad_1, atol=1e-06)
    assert torch.allclose(separate_grad_2, combined_grad_2, atol=1e-06)
    print("Gradients equal!")

    
def _register_embedding_gradient_hooks(embedding_gradients, model: Model):
    """
    Registers a backward hook on the embedding layer of the model provided.
    The embedding gradients are caught in the hook and added to the embedding
    list passed in. 
    """
    def hook_layers(module, grad_in, grad_out):
        embedding_gradients.append(grad_out[0])

    backward_hooks = []
    embedding_layer = find_embedding_layer(model)
    backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))

    return backward_hooks

def main():

    # Run tests
    # print("TEST #0: one embedding layer")
    # print("----------------------------")
    # model_1 = TestModel0()
    # model_2 = TestModel0()
    # test_embedding_layer(model_1, model_2)
    # print()

    # print("TEST #1: one linear layer")
    # print("-------------------------")
    # model_1 = TestModel1()
    # model_2 = TestModel1()
    # test_one_linear_layer(model_1, model_2)
    # print()

    # print("TEST #2: one linear layer with one layer norm")
    # print("---------------------------------------------")
    # model_1 = TestModel2()
    # model_2 = TestModel2()
    # test_one_linear_with_layer_norm(model_1, model_2)
    # print()

    # print("TEST #3: one BERT Self Attention layer")
    # print("--------------------------------------")
    # class Config:
    #     pass 
    # config = Config()
    # config.hidden_size = 9
    # config.num_attention_heads = 3
    # config.output_attentions = False
    # config.attention_probs_dropout_prob = 0

    # model_1 = BertSelfAttention(config)
    # model_2 = BertSelfAttention(config)
    # test_one_bert_self_attention_layer(model_1, model_2)
    # print()

    print("TEST #4: full BERT model")
    print("------------------------")
    reader = get_sst_reader('BERT')

    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    vocab = Vocabulary.from_instances(train_data)
    train_data.index_with(vocab)

    model_1 = get_model('BERT', vocab, True, transformer_dim=768)
    model_2 = get_model('BERT', vocab, True, transformer_dim=256)
    model_1.eval()
    model_2.eval()

    test_full_bert_model(model_1, 768, model_2, 256, train_data, vocab, cuda=True)
    print()

    print("All Tests Passed.")

if __name__ == "__main__":
    main()