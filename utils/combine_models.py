from typing import Dict, Optional

from overrides import overrides

from allennlp.models import BasicClassifier, Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator, util
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding,PretrainedTransformerEmbedder

from transformers.modeling_bert import BertSelfAttention,BertEmbeddings,BertModel

import torch

import copy 

class CombinedModel(torch.nn.Module):
    def __init__(self, combined_model):
        super(CombinedModel, self).__init__()
        self.combined_model = combined_model
        self.merge_layer = MergeLayer()

    def forward(self, *args, **kwargs):
        output = self.combined_model(*args, **kwargs)
        if not isinstance(self.combined_model, BasicClassifierCombined):
            return self.merge_layer(output)
        return output 

class MergeLayer(torch.nn.Module):
    def __init__(self):
        super(MergeLayer, self).__init__()

    def forward(self, x):
        """
        Squeezes the double width input back to single width input.
        NOTE: the current possible shapes coming in that I could think of are:
            1) vector of size (*,)
                dimension 1 marks the embedding dimension
            2) 2D matrix of size (*, *): 
                dimension 1 marks the number of sentences, 
                dimension 2 marks the embedding dimension
            3) 3D matrix of size (*, *, *):
                dimension 1 marks the batch number
                dimension 2 marks the number of sentences in current batch
                dimension 3 marks the embedding dimension
        """
        # BertSelfAttention returns a tuple, so when testing
        # just this layer, we need to take the first element
        # (which is the element that contains the logits)
        if type(x) == tuple:
            x = x[0]
        if len(x.shape) == 1:
            return x[:int(len(x)/2)] + x[int(len(x)/2):]
        elif len(x.shape) == 2:
            result = torch.empty((x.shape[0], int(x.shape[1]/2)))
            for i, row in enumerate(x):
                result[i] = x[i][:int(len(x[i])/2)] + x[i][int(len(x[i])/2):]

            if x.is_cuda:
                # print("********")
                result = result.cuda()
            return result
        elif len(x.shape) == 3:
            result = torch.empty((x.shape[0], x.shape[1], int(x.shape[2]/2)))
            for i, row in enumerate(x):
                for j, col in enumerate(x[i]):
                    result[i][j] = x[i][j][:int(len(x[i][j])/2)] + x[i][j][int(len(x[i][j])/2):]
            
            if x.is_cuda:
                # print("********")
                result = result.cuda()
            return result

class DoubleLayerNorm(torch.nn.Module):
    """
    Replaces PyTorch's built-in LayerNorm for a model that is twice as wide.
    The built-in LayerNorm would take the mean and standard deviation for 
    the entire wide input. This is wrong for the wide model, and instead
    we take the mean and standard deviation for each half of the input. 
    """
    def __init__(
        self, 
        normalized_shape_1,
        normalized_shape_2, 
        weight_1=None, 
        weight_2=None, 
        bias_1=None, 
        bias_2=None,
        eps_1=1e-5,
        eps_2=1e-5,
        elementwise_affine_1=True,
        elementwise_affine_2=True
    ):
        super(DoubleLayerNorm, self).__init__()
        self.layer_norm_l = torch.nn.LayerNorm(normalized_shape_1, eps=eps_1, elementwise_affine=elementwise_affine_1)
        self.layer_norm_r = torch.nn.LayerNorm(normalized_shape_2, eps=eps_2, elementwise_affine=elementwise_affine_2)

        self.layer_norm_l.weight = weight_1
        self.layer_norm_l.bias = bias_1 

        self.layer_norm_r.weight = weight_2
        self.layer_norm_r.bias = bias_2 

    def forward(self, x):
        x_l, x_r = torch.split(x, (self.layer_norm_l.normalized_shape[0], self.layer_norm_r.normalized_shape[0]), dim=-1)
        return torch.cat((self.layer_norm_l(x_l), self.layer_norm_r(x_r)), dim=-1)

class BasicClassifierCombined(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.
    Registered as a `Model` with name "basic_classifier".
    # Parameters
    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = "labels")
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    NOTE: BasicClassifierCombined is very similar to the original BasicClassifier class.
    The only difference is that before softmax, the logits are squeezed into a single model 
    dimension again (since the logits coming in are twice as wide as for a regular model). 
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tags",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._merge_layer = MergeLayer()
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : TextFieldTensors
            From a `TextField`
        label : torch.IntTensor, optional (default = None)
            From a `LabelField`
        # Returns
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # print("ENTERING COMBINED FORWARD")
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)


        logits = self._classification_layer(embedded_text)
        # print("combined model\n",logits)
        # logits = self._merge_layer(logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        # print("ENTERING COMBINED MAKE OUTPUT HUMAN READABLE")
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        token_id.item(),namespace=self._namespace
                    )
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        # print("OUTPUT DICT", output_dict)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics 

def merge_models(model_1, model_2,task="sst"):
    """
    Takes in two models with the same layers but different parameters
    and merges them into one wider model. 
    """
    def _merge_models(model_1, model_2):
        
        result_model = copy.deepcopy(model_1)

        if isinstance(model_1, BasicClassifier):
            if task != "rc":
                result_model = _add_basic_classifier_combined(model_1, model_2)


        if isinstance(model_1, torch.nn.Embedding):
            # print(model_1,model_2)
            result_model = _add_embedding_layer(model_1, model_2)
            # print(result_model)
        elif isinstance(model_1, torch.nn.Linear):
            result_model = _add_linear_layer(model_1, model_2)

        elif isinstance(model_1, torch.nn.LayerNorm):
            result_model = _add_double_norm_layer(model_1, model_2)

        elif isinstance(model_1, BertSelfAttention):
            result_model = _add_bert_self_attention_layer(model_1, model_2)

        for name_1, name_2 in zip(model_1._modules, model_2._modules):
            module_1 = model_1._modules[name_1]
            module_2 = model_2._modules[name_2]
            
            result_model._modules[name_1] = _merge_models(module_1, module_2)
        # if isinstance(model_1, TextFieldEmbedder):
        #     print("---",result_model._modules["token_embedder_tokens"])
        #     print(result_model._token_embedders["tokens"])
        #     print("---")
        return result_model

    result_model = _merge_models(model_1, model_2)
    # return CombinedModel(result_model)
    result_model._text_field_embedder._token_embedders["tokens"].output_dim = 1024
    # print(result_model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings)
    if task == "rc":
        result_model._linear_layer = _add_final_linear_layer(model_1._linear_layer, model_2._linear_layer)
    else:
        result_model._classification_layer = _add_final_linear_layer(model_1._classification_layer, model_2._classification_layer)
    return result_model

def _add_basic_classifier_combined(model_1, model_2):
    """
    Returns a BasicClassifierCombined classifier based on 
    model_1 and model_2. 
    """
    return BasicClassifierCombined(
        vocab=model_1.vocab,
        text_field_embedder=model_1._text_field_embedder,
        seq2vec_encoder=model_1._seq2vec_encoder,
        feedforward=model_1._feedforward,
        dropout=model_1._dropout.p,
        num_labels=model_1._num_labels
    )

def _add_bert_self_attention_layer(model_1, model_2):
    """
    Returns a BertSelfAttention2 layer based on model_1 
    and model_2.
    """
    class Config:
        pass 

    config = Config()
    config.hidden_size = model_1.all_head_size + model_2.all_head_size
    config.num_attention_heads = model_1.num_attention_heads + model_2.num_attention_heads
    config.output_attentions = model_1.output_attentions
    # print(model_1.output_attentions,model_2.output_attentions)
    config.attention_probs_dropout_prob = model_1.dropout.p

    return BertSelfAttention(config)

def _add_double_norm_layer(model_1, model_2):
    """
    Returns a DoubleLayerNorm which combines the built-in
    LayerNorm of model_1 and model_2.
    """
    return DoubleLayerNorm(
        model_1.normalized_shape,
        model_2.normalized_shape,
        model_1.weight, 
        model_2.weight, 
        model_1.bias, 
        model_2.bias, 
        eps_1=model_1.eps,
        eps_2=model_2.eps,
        elementwise_affine_1=model_1.elementwise_affine,
        elementwise_affine_2=model_2.elementwise_affine
    )

def _add_embedding_layer(model_1, model_2):
    """
    Returns an embedding layer with a weight matrix
    of the follwing structure:
        [MODEL_1 EMBEDDING MATRIX ; MODEL_2 EMBEDDING MATRIX]
    """
    result_layer = torch.nn.Embedding(model_1.num_embeddings, model_1.embedding_dim+model_2.embedding_dim)
    result_layer.weight = torch.nn.Parameter(torch.cat((model_1.weight.data, model_2.weight.data), dim=1))
    return result_layer

def _add_linear_layer(model_1, model_2):
    """
    Returns a linear layer that has the following 
    weight structure: 
        [ MODEL_1 WEIGHT  000000000000000]
        [ 000000000000000  MODEL_2 WEIGHT]
    """
    data_1 = model_1.weight.data
    data_2 = model_2.weight.data 
    top_right = torch.zeros((data_1.size()[0],data_2.size()[1]))
    # print(top_right)
    torch.nn.init.normal_(top_right,mean=0,std=1e-07)
    # print(top_right)
    if data_1.is_cuda:
        top_right = top_right.cuda()
    new_weight_top = torch.cat((data_1, top_right), dim=1)

    bottom_left = torch.zeros((data_2.size()[0], data_1.size()[1]))
    torch.nn.init.normal_(bottom_left,mean=0,std=1e-07)
    if data_1.is_cuda:
        bottom_left = bottom_left.cuda()
    new_weight_bottom = torch.cat((bottom_left, data_2), dim=1)
    new_weight = torch.cat((new_weight_top, new_weight_bottom), dim=0)
    
    new_bias = torch.cat((model_1.bias, model_2.bias), dim=0)

    result_model = torch.nn.Linear(model_1.in_features+model_2.in_features, model_1.out_features+model_2.out_features)
    result_model.weight = torch.nn.Parameter(new_weight)
    result_model.bias = torch.nn.Parameter(new_bias)

    return result_model

def _add_final_linear_layer(model_1, model_2):
    """
    Returns a linear layer that has the following 
    weight structure: 
        [ MODEL_1 WEIGHT MODEL_2 WEIGHT]
    """
    data_1 = model_1.weight.data
    data_2 = model_2.weight.data 
    # print(data_1.size(),data_2.shape)

    new_weight = torch.cat((data_1, data_2), dim=1)
    new_bias = model_1.bias + model_2.bias
    print(new_weight.size(),new_bias.size())
    result_model = torch.nn.Linear(model_1.in_features+model_2.in_features, model_1.out_features)
    result_model.weight = torch.nn.Parameter(new_weight)
    result_model.bias = torch.nn.Parameter(new_bias)

    return result_model