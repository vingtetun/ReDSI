from transformers import AutoTokenizer

from redsi.formats import formats
from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "Tokenize text fields into input_ids. "
    "If text_tokens already exists and is non-empty, keep it. "
    "Applies to every split containing text and docid fields. "
    "Optionally rewrites or drops text after tokenization to reduce memory."
)

PLUGIN_PARAMS = [
    Param(
        "tokenizer_name",
        "str",
        default="google-t5/t5-base",
        help="Hugging Face tokenizer name or path.",
    ),
    Param(
        "document_max_length",
        "int",
        default=32,
        help="Maximum number of tokens for document texts (excluding the appended EOS token).",
    ),
    Param(
        "query_max_length",
        "int",
        default=128,
        help="Maximum number of tokens for query texts (excluding the appended EOS token).",
    ),
    Param(
        "require_single_token_per_docid_char",
        "bool",
        default=False,
        help=(
            "If True, each character of the document identifier is tokenized "
            "individually and must map to exactly one tokenizer token id. "
            "Raises an error if any character produces zero or multiple tokens. "
            "This enforces 1 character -> 1 token for docids using the model tokenizer."
        ),
    ),
    Param(
        "rewrite_text_from_text_tokens",
        "bool",
        default=False,
        help=(
            "If True, rewrite text from the final text_tokens after tokenization. "
            "Useful to compact text to the exact token-limited view."
        ),
    ),
    Param(
        "drop_text_after_tokenization",
        "bool",
        default=True,
        help=("If True, drop text after building text_tokens. Useful when downstream only needs tokenized inputs."),
    ),
]

ERR_FAIL_ON_TRUNCATION = "Tokenization truncated examples at max_length={max_length} ({current_length})"
ERR_FAIL_TO_TOKENIZE = "Character {ch!r} does not map to exactly one token."
ERR_NO_EOS_TOKEN = "Tokenizer has no eos_token_id."
ERR_DUPLICATE_ADDED_DOCID_TOKEN = "tokenizer: duplicate token {token!r} in tokens split"
ERR_INCONSISTENT_BASE_VOCAB_SIZE = (
    "tokenizer: inconsistent base_vocab_size values in tokens split ({first} != {current})"
)

TOKENIZER_FIELD_INPUT_IDS = "input_ids"
TOKENS_SPLIT_NAME = "tokens"
FEATURE_BASE_VOCAB_SIZE = "base_vocab_size"
FEATURE_TOKENS = "tokens"


class TokenizerPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params):
        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_name, use_fast=True)
        return self._tokenize_all_splits(ds, params, tokenizer)

    def _tokenize_all_splits(self, ds, params, tokenizer):
        drop_text_after_tokenization = bool(params.drop_text_after_tokenization)
        added_docid_token_ids = _read_added_docid_token_ids(ds)

        for split_key in list(ds.keys()):
            split_ds = ds[split_key]
            column_names = set(split_ds.column_names)

            required = {formats.FEATURE_TEXT, formats.FEATURE_DOCID}
            if not required.issubset(column_names):
                continue

            default_kind = self._default_kind_for_split(split_key)

            fn_kwargs = dict(
                tokenizer=tokenizer,
                document_max_length=params.document_max_length,
                query_max_length=params.query_max_length,
                default_kind=default_kind,
                require_single_token_per_docid_char=params.require_single_token_per_docid_char,
                rewrite_text_from_text_tokens=bool(params.rewrite_text_from_text_tokens),
                drop_text_after_tokenization=drop_text_after_tokenization,
                added_docid_token_ids=added_docid_token_ids,
            )

            ds[split_key] = split_ds.map(
                tokenize_batch,
                desc=f"Tokenize {split_key}",
                batched=True,
                fn_kwargs=fn_kwargs,
                remove_columns=[formats.FEATURE_TEXT] if drop_text_after_tokenization else None,
            )

        return ds

    def _default_kind_for_split(self, split_key):
        if split_key == formats.KEY_DOCUMENTS:
            return formats.FEATURE_KIND_DOCUMENT
        return formats.FEATURE_KIND_QUERY


def _has_non_empty_token_list(value):
    return isinstance(value, list) and len(value) > 0


def _ensure_eos(ids, eos_id):
    if eos_id is None:
        raise ValueError(ERR_NO_EOS_TOKEN)

    ids = list(ids)
    if not ids or ids[-1] != eos_id:
        ids.append(eos_id)
    return ids


def _read_added_docid_token_ids(ds):
    if TOKENS_SPLIT_NAME not in ds:
        return {}

    tokens_ds = ds[TOKENS_SPLIT_NAME]
    required_columns = {FEATURE_BASE_VOCAB_SIZE, FEATURE_TOKENS}
    if not required_columns.issubset(set(tokens_ds.column_names)):
        return {}

    token_ids = {}
    base_vocab_size = None

    for row in tokens_ds:
        row_base_vocab_size = int(row[FEATURE_BASE_VOCAB_SIZE])
        if base_vocab_size is None:
            base_vocab_size = row_base_vocab_size
        elif base_vocab_size != row_base_vocab_size:
            raise ValueError(
                ERR_INCONSISTENT_BASE_VOCAB_SIZE.format(
                    first=base_vocab_size,
                    current=row_base_vocab_size,
                )
            )

        for token in row[FEATURE_TOKENS] or []:
            if token in token_ids:
                raise ValueError(ERR_DUPLICATE_ADDED_DOCID_TOKEN.format(token=token))
            token_ids[token] = base_vocab_size + len(token_ids)

    return token_ids


def _build_docid_token_ids(
    docids,
    tokenizer,
    require_single_token_per_docid_char,
    eos_id,
    added_docid_token_ids=None,
):
    if require_single_token_per_docid_char:
        vocab = tokenizer.get_vocab()
        fallback_ids = [[encode_char_as_single_token_id(tokenizer, vocab, ch) for ch in docid] for docid in docids]
    else:
        encoded = tokenizer(
            docids,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            verbose=False,
        )
        fallback_ids = encoded[TOKENIZER_FIELD_INPUT_IDS]

    added_docid_token_ids = added_docid_token_ids or {}
    out = []
    for docid, ids in zip(docids, fallback_ids):
        token_id = added_docid_token_ids.get(docid)
        if token_id is not None:
            ids = [int(token_id)]
        out.append(_ensure_eos(ids, eos_id))
    return out


def _tokenize_missing_texts(texts, tokenizer, max_length, fail_on_truncation, eos_id):
    tokenizer_kwargs = dict(
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_length,
        verbose=False,
    )
    if fail_on_truncation:
        tokenizer_kwargs["return_overflowing_tokens"] = True
        tokenizer_kwargs["stride"] = 0

    texts_tokens = tokenizer(texts, **tokenizer_kwargs)

    input_ids_all = texts_tokens[TOKENIZER_FIELD_INPUT_IDS]
    mapping = texts_tokens.get("overflow_to_sample_mapping")
    batch_size = len(texts)

    if mapping is not None:
        first_chunk = [None] * batch_size
        chunks_per_sample = [0] * batch_size

        for ids, sample_idx in zip(input_ids_all, mapping):
            chunks_per_sample[sample_idx] += 1
            if first_chunk[sample_idx] is None:
                first_chunk[sample_idx] = ids

        if fail_on_truncation and any(c > 1 for c in chunks_per_sample):
            index = next(i for i, c in enumerate(chunks_per_sample) if c > 1)
            full_tokens = tokenizer(
                texts[index],
                add_special_tokens=False,
                return_token_type_ids=False,
                return_attention_mask=False,
                verbose=False,
            )
            current_length = len(full_tokens[TOKENIZER_FIELD_INPUT_IDS])
            raise ValueError(
                ERR_FAIL_ON_TRUNCATION.format(
                    max_length=max_length,
                    current_length=current_length,
                )
            )

        return [_ensure_eos(ids or [], eos_id) for ids in first_chunk]

    if fail_on_truncation and len(input_ids_all) > batch_size:
        raise ValueError(
            ERR_FAIL_ON_TRUNCATION.format(
                max_length=max_length,
                current_length="unknown",
            )
        )

    return [_ensure_eos(ids, eos_id) for ids in input_ids_all[:batch_size]]


def _text_tokenization_spec(kind, document_max_length, query_max_length):
    if kind == formats.FEATURE_KIND_DOCUMENT:
        return document_max_length, False

    return query_max_length, True


def _rewrite_texts_from_token_ids(tokenizer, token_ids_batch):
    out = []
    for ids in token_ids_batch:
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out.append(text)
    return out


def tokenize_batch(
    batch,
    tokenizer,
    document_max_length,
    query_max_length,
    default_kind,
    require_single_token_per_docid_char,
    rewrite_text_from_text_tokens,
    drop_text_after_tokenization,
    added_docid_token_ids,
):
    batch = dict(batch)
    texts = batch[formats.FEATURE_TEXT]
    kinds = batch.get(formats.FEATURE_KIND)
    existing_text_tokens = batch.get(formats.FEATURE_TEXT_TOKENS)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(ERR_NO_EOS_TOKEN)

    batch_size = len(texts)

    if existing_text_tokens is None:
        existing_text_tokens = [None] * batch_size

    texts_token_ids = [None] * batch_size
    missing_by_spec = {}

    for i in range(batch_size):
        current_ids = existing_text_tokens[i]
        if _has_non_empty_token_list(current_ids):
            texts_token_ids[i] = _ensure_eos(current_ids, eos_id)
        else:
            kind = kinds[i] if kinds is not None else default_kind
            spec = _text_tokenization_spec(
                kind=kind,
                document_max_length=document_max_length,
                query_max_length=query_max_length,
            )
            indices, group_texts = missing_by_spec.setdefault(spec, ([], []))
            indices.append(i)
            group_texts.append(texts[i])

    for (max_length, fail_on_truncation), (missing_indices, missing_texts) in missing_by_spec.items():
        new_token_ids = _tokenize_missing_texts(
            texts=missing_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            fail_on_truncation=fail_on_truncation,
            eos_id=eos_id,
        )
        for idx, ids in zip(missing_indices, new_token_ids):
            texts_token_ids[idx] = ids

    batch[formats.FEATURE_TEXT_TOKENS] = texts_token_ids

    if drop_text_after_tokenization:
        batch.pop(formats.FEATURE_TEXT, None)
    elif rewrite_text_from_text_tokens:
        batch[formats.FEATURE_TEXT] = _rewrite_texts_from_token_ids(
            tokenizer=tokenizer,
            token_ids_batch=texts_token_ids,
        )

    docids = batch[formats.FEATURE_DOCID]
    existing_docid_tokens = batch.get(formats.FEATURE_DOCID_TOKENS)

    if existing_docid_tokens is None:
        existing_docid_tokens = [None] * batch_size

    built_docid_tokens = None
    if any(not _has_non_empty_token_list(v) for v in existing_docid_tokens):
        built_docid_tokens = _build_docid_token_ids(
            docids=docids,
            tokenizer=tokenizer,
            require_single_token_per_docid_char=require_single_token_per_docid_char,
            eos_id=eos_id,
            added_docid_token_ids=added_docid_token_ids,
        )

    final_docid_tokens = []
    for i, existing in enumerate(existing_docid_tokens):
        if _has_non_empty_token_list(existing):
            final_docid_tokens.append(_ensure_eos(existing, eos_id))
        else:
            final_docid_tokens.append(built_docid_tokens[i])

    batch[formats.FEATURE_DOCID_TOKENS] = final_docid_tokens

    return batch


def encode_char_as_single_token_id(tokenizer, vocab, ch):
    if ch in vocab:
        return vocab[ch]
    raise ValueError(ERR_FAIL_TO_TOKENIZE.format(ch=ch))
