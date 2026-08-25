import datasets
from transformers import AutoTokenizer

from redsi.formats import formats
from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "Prepend task prefixes to text_tokens based on kind. "
    "Document rows receive 'index: ' and query rows receive 'query: '."
)

PLUGIN_PARAMS = [
    Param(
        "tokenizer_name",
        "str",
        default="google-t5/t5-base",
        help="Hugging Face tokenizer name or path used to tokenize task prefixes.",
    ),
]

DOCUMENT_PREFIX = "index:"
QUERY_PREFIX = "query:"

ERR_INVALID_DATASET = "task_prefix_tokens: expected a DatasetDict, got {dataset_type}"
ERR_NO_EOS_TOKEN = "Tokenizer has no eos_token_id."

TOKENIZER_FIELD_INPUT_IDS = "input_ids"


class TaskPrefixTokensPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params):
        if not isinstance(ds, datasets.DatasetDict):
            raise TypeError(ERR_INVALID_DATASET.format(dataset_type=type(ds)))

        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name, use_fast=True)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(ERR_NO_EOS_TOKEN)

        fn_kwargs = {
            "document_prefix_tokens": _encode_prefix(tokenizer, DOCUMENT_PREFIX),
            "query_prefix_tokens": _encode_prefix(tokenizer, QUERY_PREFIX),
            "eos_id": eos_id,
        }

        out = dict(ds)
        for split_name, split_ds in ds.items():
            column_names = set(split_ds.column_names)
            required = {formats.FEATURE_KIND, formats.FEATURE_TEXT_TOKENS}
            if not required.issubset(column_names):
                continue

            out[split_name] = split_ds.map(
                add_task_prefixes_batch,
                desc=f"Add task prefixes {split_name}",
                batched=True,
                fn_kwargs=fn_kwargs,
            )

        return datasets.DatasetDict(out)


def _encode_prefix(tokenizer, prefix):
    encoded = tokenizer(
        prefix,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        verbose=False,
    )
    return list(encoded[TOKENIZER_FIELD_INPUT_IDS])


def add_task_prefixes_batch(
    batch,
    document_prefix_tokens,
    query_prefix_tokens,
    eos_id,
):
    batch = dict(batch)
    kinds = batch[formats.FEATURE_KIND]
    text_tokens = batch[formats.FEATURE_TEXT_TOKENS]

    batch[formats.FEATURE_TEXT_TOKENS] = [
        _add_task_prefix(
            kind=kind,
            token_ids=token_ids,
            document_prefix_tokens=document_prefix_tokens,
            query_prefix_tokens=query_prefix_tokens,
            eos_id=eos_id,
        )
        for kind, token_ids in zip(kinds, text_tokens)
    ]

    return batch


def _add_task_prefix(
    kind,
    token_ids,
    document_prefix_tokens,
    query_prefix_tokens,
    eos_id,
):
    token_ids = list(token_ids or [])

    if kind == formats.FEATURE_KIND_DOCUMENT:
        prefix_tokens = document_prefix_tokens
    elif kind == formats.FEATURE_KIND_QUERY:
        prefix_tokens = query_prefix_tokens
    else:
        return token_ids

    if token_ids and token_ids[-1] == eos_id:
        token_ids = token_ids[:-1]

    return list(prefix_tokens) + token_ids + [eos_id]


plugin = TaskPrefixTokensPlugin()
