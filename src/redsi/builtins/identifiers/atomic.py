import datasets
from transformers import AutoTokenizer

from redsi.formats import formats
from redsi.formats.results import IdentifierResult
from redsi.plugins import Param

PLUGIN_DESCRIPTION = "Assign simple atomic document identifiers with a fixed prefix and zero-padded index."

TOKENS_SPLIT_NAME = "tokens"
FEATURE_BASE_VOCAB_SIZE = "base_vocab_size"
FEATURE_TOKENS = "tokens"

PLUGIN_PARAMS = [
    Param(
        "prefix",
        "str",
        default="doc",
        help="Identifier prefix used inside angle brackets.",
    ),
    Param(
        "start",
        "int",
        default=0,
        help="First numeric identifier value.",
    ),
    Param(
        "tokenizer_name",
        "str",
        default="google-t5/t5-base",
        help="Hugging Face tokenizer name used to infer the base vocabulary size.",
    ),
]

ERR_START_NEGATIVE = "atomic: start ({start}) must be >= 0"


class AtomicIdentifierPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, documents, params, seed):
        del seed

        documents = list(documents)
        width = infer_identifier_width(
            num_documents=len(documents),
            start=params.start,
        )

        base_vocab_size = infer_base_vocab_size(params.tokenizer_name)

        docid_map = {}
        tokens = []
        for idx, doc in enumerate(documents, start=params.start):
            token = format_identifier(params.prefix, idx, width)
            docid_map[doc[formats.FEATURE_DOCID]] = token
            tokens.append(token)

        tokens_ds = datasets.Dataset.from_dict(
            {
                FEATURE_BASE_VOCAB_SIZE: [base_vocab_size],
                FEATURE_TOKENS: [tokens],
            }
        )

        return IdentifierResult(
            docid_map=docid_map,
            extra_splits={TOKENS_SPLIT_NAME: tokens_ds},
        )


def format_identifier(prefix, idx, width):
    return f"<{prefix}_{idx:0{width}d}>"


def infer_identifier_width(num_documents, start):
    if start < 0:
        raise ValueError(ERR_START_NEGATIVE.format(start=start))
    if num_documents <= 0:
        return 1

    max_index = int(start) + int(num_documents) - 1
    return len(str(max_index))


def infer_base_vocab_size(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return len(tokenizer)
