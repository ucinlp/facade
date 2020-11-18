from typing import Any, Tuple

import torch 

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, SpacyTokenizer
from allennlp.models import BasicClassifier, Model
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, Embedding, PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder, ClsPooler
from allennlp.modules import FeedForward
from allennlp_models.rc.transformer_qa import TransformerSquadReader, TransformerQA

def get_bert_model(
    vocab: Vocabulary, 
    transformer_dim: int, 
    model_name: str, 
    num_layers: int, 
    activations, 
    dropout: int,
    num_labels:int=None,
    mismatched: bool=False
):
    """
    Construct and return a bert model with the given configuration
    parameters.
    """
    if mismatched:
        token_embedder = PretrainedTransformerMismatchedEmbedder(
            model_name=model_name, hidden_size=transformer_dim
        )
    else: 
        token_embedder = PretrainedTransformerEmbedder(
            model_name=model_name, hidden_size=transformer_dim
        )
    text_field_embedders = BasicTextFieldEmbedder({ "tokens": token_embedder })
    seq2vec_encoder = ClsPooler(embedding_dim=transformer_dim)
    feedforward = FeedForward(
        input_dim=transformer_dim, 
        num_layers=num_layers, 
        hidden_dims=transformer_dim, 
        activations=activations
    )
    dropout = dropout
    
    bert_model = BasicClassifier(
        vocab=vocab, 
        text_field_embedder=text_field_embedders, 
        seq2vec_encoder=seq2vec_encoder,
        feedforward=feedforward,
        dropout=dropout,
        num_labels=num_labels
    )

    return bert_model

def get_model(model_name: str, vocab: Vocabulary, cuda: bool, num_labels: int=None, transformer_dim: int=768, task: str=None) -> Any: 
    """
    Construct model based on the model_name passed in. Load model to cuda if
    cuda equals True. 
    """

    if model_name == 'LSTM':
        print('lstm')
        word_embeddings, word_embedding_dim = get_embedding_config(vocab, "glove")
        encoder = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                word_embedding_dim,
                hidden_size=512,
                num_layers=2,
                batch_first=True
            )
        )
        model = BasicClassifier(vocab, word_embeddings, encoder)

    elif model_name == 'BERT':
        if task == "QA":
            model = TransformerQA(vocab=vocab, transformer_model_name='bert-base-cased', hidden_size=transformer_dim)
        else: 
            model = get_bert_model(vocab, transformer_dim, "bert-base-uncased", 1, torch.nn.Tanh(), 0.1, num_labels)

    if cuda: 
        model.cuda()

    return model 

def get_embedding_config(vocab: Vocabulary, embedding_type: str) -> Tuple[BasicTextFieldEmbedder, int]:
    """
    Return the appropriate embedder and embedding dimension based on 
    the embedding_type passed in. 
    """
    # Randomly initialize vectors
    if embedding_type == None:
        # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        # word_embedding_dim = 300
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=10
        )
        word_embedding_dim = 10

    # Load word2vec vectors
    elif embedding_type == "glove":
        embedding_path = "embeddings/glove.840B.300d.txt"
        weight = _read_pretrained_embeddings_file(
            embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"
        )
        token_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"),
            embedding_dim=300,
            weight=weight,
            trainable=True
        )
        word_embedding_dim = 300

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    return word_embeddings, word_embedding_dim

def load_model(model: Model, file: str, task="sst") -> None:
    """
    Load model weights into model using the weights stored
    in the location of file parameter. 
    """
    with open(file, "rb") as f:
        loaded = torch.load(f)
        new_dict = loaded.copy()
        if task == "QA":
            keys = loaded.keys()
            for layer in keys:
                if layer == "_classification_layer.bias":
                    new_dict["_linear_layer.bias"] = loaded[
                        "_classification_layer.bias"
                    ]
                    del new_dict["_classification_layer.bias"]
                if layer == "_classification_layer.weight":
                    new_dict["_linear_layer.weight"] = loaded[
                        "_classification_layer.weight"
                    ]
                    del new_dict["_classification_layer.weight"]
                if layer == "_feedforward._linear_layers.0.weight":
                    del new_dict["_feedforward._linear_layers.0.weight"]
                if layer == "_feedforward._linear_layers.0.bias":
                    del new_dict["_feedforward._linear_layers.0.bias"]
                if (
                    layer
                    == "_text_field_embedder.token_embedder_tokens.transformer_model.embeddings.word_embeddings.weight"
                ):
                    print(loaded[layer][:28996, :].shape)
                    # model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings = model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings.weight[:28996,:]
                    # print(model._text_field_embedder._token_embedders["tokens"].transformer_model.embeddings.word_embeddings.weight.size())
                    new_dict[layer] = loaded[layer][:28996, :]

        # print(new_dict.keys())
        model.load_state_dict(new_dict)

def get_sst_reader(
    model_name: str, use_subtrees=True
) -> StanfordSentimentTreeBankDatasetReader:
    """
    Constructs and returns a SST Dataset Reader based on the model name. 
    """
    # load the binary SST dataset.
    if model_name == "BERT":
        bert_indexer = PretrainedTransformerIndexer("bert-base-uncased")
        tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class",
            use_subtrees=use_subtrees,
            token_indexers={"tokens": bert_indexer},
            tokenizer=tokenizer
        )
    else: 
        single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class",
            use_subtrees=use_subtrees,
            token_indexers={"tokens": single_id_indexer}
        )

    return reader 

def get_snli_reader(model_name: str, combined_input_fields: bool=True) -> SnliReader:
    """
    Constructs and returns a SNLI Dataset Reader based on the model name. 
    """
    if model_name == "BERT":
        bert_indexer = PretrainedTransformerIndexer("bert-base-uncased")
        tokenizer = PretrainedTransformerTokenizer(model_name="bert-base-uncased")
        reader = SnliReader(
            token_indexers={"tokens": bert_indexer}, 
            tokenizer=tokenizer, 
            combine_input_fields=combined_input_fields
        )
    else:
        single_id_indexer = SingleIdTokenIndexer(
            lowercase_tokens=True
        ) # word tokenizer
        tokenizer = SpacyTokenizer(end_tokens=["@@NULL@@"])
        reader = SnliReader(
            token_indexers={"tokens": single_id_indexer}, tokenizer=tokenizer
        )

    return reader 

def get_qa_reader(model_name: str, combined_input_fields: bool=True) -> SnliReader:
    """
    Constructs and returns a SQUAD Dataset Reader based on the model name. 
    """
    if model_name == "BERT":
        reader = TransformerSquadReader(transformer_model_name='bert-base-cased')
    else:
        pass 

    return reader 

def save_model(
    model: Model, vocab: Vocabulary, folder: str, model_path: str, vocab_path: str
):
    try:
        os.mkdir(folder)
    except:
        print("directory already created")
    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(vocab_path)