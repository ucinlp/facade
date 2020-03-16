import logging
from typing import Dict

from allennlp.data.dataset_readers import SnliReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_snli")
class BertSnliReader(SnliReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    and "tokens".
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super(BertSnliReader, self).__init__(tokenizer, token_indexers, lazy)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        # Here, we join the premise with the hypothesis, dropping the CLS token from the hypothesis.        
        # This gives us our desired inputs: "[CLS] premise [SEP] hypothesis [SEP]"
        tokens = premise_tokens + hypothesis_tokens[1:]
        fields["tokens"] = TextField(tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)