import hashlib

from . import features, pageids, titles

# Extra keys
KEY_FULL_TEXT_UNCASED_MD5 = "document_text_uncased_md5"
KEY_FULL_TEXT_CASED_MD5 = "document_text_cased_md5"

# Deduplication methods
DEDUP_METHOD_NONE = "none"
DEDUP_METHOD_FULL_TEXT_UNCASED = "full-text"
DEDUP_METHOD_FULL_TEXT_CASED = "full-text-cased"
DEDUP_METHOD_URL_UNCASED = "url"
DEDUP_METHOD_URL_CASED = "url-cased"
DEDUP_METHOD_TITLE_UNCASED = "title"
DEDUP_METHOD_TITLE_CASED = "title-cased"
DEDUP_METHOD_NCITITLE = "ncititle"
DEDUP_METHOD_NCI_H1 = "nci-h1"
DEDUP_METHOD_H1_UNCASED = "h1"
DEDUP_METHOD_H1_CASED = "h1-cased"
DEDUP_METHOD_PAGEID = "pageid"
DEDUP_METHOD_TEXT4K = "text4k"
DEDUP_METHOD_TEXT4K_CASED = "text4k-cased"
DEDUP_METHOD_TOKENS_32 = "tokens-32"
DEDUP_METHOD_TOKENS_32_CASED = "tokens-32-cased"

# Errors
ERR_UNSUPPORTED_DEDUP_METHOD = "The deduplication method {name} is unsupported."
ERR_MISSING_NCITITLE_DEDUP_TITLE = (
    "ncititle deduplication requires '{field}' computed before text formatting. "
    "Regenerate the Natural Questions simplified cache for this text style."
)

_TOKENIZER = None
_T5_TOKENIZER = None


def setup_tokenizer():
    global _TOKENIZER
    from transformers import BertTokenizer

    _TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


def setup_t5_tokenizer():
    global _T5_TOKENIZER
    from transformers import T5TokenizerFast

    _T5_TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")


def get_hash_for_url_uncased(example):
    return get_hash_for_url(example, True)


def get_hash_for_title_uncased(example):
    return get_hash_for_title(example, True)


def get_hash_for_ncititle(example):
    text = get_ncititle_dedup_title(example)
    return get_hash_for(example, normalize_nci(text))


def get_hash_for_nci_h1(example):
    text = get_h1_title(example)
    return get_hash_for(example, normalize_nci(text))


def get_ncititle_dedup_title(example):
    if features.KEY_NCI_DEDUP_TITLE in example:
        return example.get(features.KEY_NCI_DEDUP_TITLE) or ""

    original_format = example.get(features.KEY_ORIGINAL_FORMAT)
    if original_format == features.KEY_ORIGINAL_FORMAT_FULL:
        return titles.from_url(example[features.KEY_DOCUMENT_URL])

    text = get_h1_title(example, raise_if_missing=False)
    if text:
        return text

    raise ValueError(ERR_MISSING_NCITITLE_DEDUP_TITLE.format(field=features.KEY_NCI_DEDUP_TITLE))


def get_h1_title(example, raise_if_missing=True):
    if features.KEY_DOCUMENT_H1_TITLE in example:
        return example.get(features.KEY_DOCUMENT_H1_TITLE) or ""

    text = titles.from_h1(example[features.KEY_DOCUMENT_TEXT])
    if text or not raise_if_missing:
        return text

    raise ValueError(ERR_MISSING_NCITITLE_DEDUP_TITLE.format(field=features.KEY_DOCUMENT_H1_TITLE))


def normalize_nci(text):
    tokens = _TOKENIZER.tokenize(text)
    token_ids = _TOKENIZER.convert_tokens_to_ids(tokens)
    text = _TOKENIZER.decode(token_ids)
    return text


def get_hash_for_h1_uncased(example):
    return get_hash_for_h1(example, True)


def get_hash_for_text4k_uncased(example):
    return get_hash_for_text4k(example, True)


def get_hash_for_full_text_uncased(example):
    return get_hash_for_full_text(example, True)


def get_hash_for_tokens_32_uncased(example):
    return get_hash_for_tokens_32(example, True)


def get_hash_for_title_cased(example):
    return get_hash_for_title(example, False)


def get_hash_for_full_text_cased(example):
    return get_hash_for_full_text(example, False)


def get_hash_for_url_cased(example):
    return get_hash_for_url(example, False)


def get_hash_for_h1_cased(example):
    return get_hash_for_h1(example, False)


def get_hash_for_text4k_cased(example):
    return get_hash_for_text4k(example, False)


def get_hash_for_tokens_32_cased(example):
    return get_hash_for_tokens_32(example, False)


def get_hash_for_url(example, uncased):
    return get_hash_for(example, example[features.KEY_DOCUMENT_URL], uncased)


def get_hash_for_title(example, uncased):
    title = titles.from_url(example[features.KEY_DOCUMENT_URL])
    return get_hash_for(example, title, uncased)


def get_hash_for_h1(example, uncased):
    title = titles.from_h1(example[features.KEY_DOCUMENT_TEXT])
    return get_hash_for(example, title, uncased)


def get_hash_for_text4k(example, uncased):
    text = example[features.KEY_DOCUMENT_TEXT][:4000]
    return get_hash_for(example, text, uncased)


def get_hash_for_full_text(example, uncased):
    key = KEY_FULL_TEXT_UNCASED_MD5 if uncased else KEY_FULL_TEXT_CASED_MD5
    if key in example:
        return example[key]

    return get_hash_for(example, example[features.KEY_DOCUMENT_TEXT], uncased)


def get_hash_for_tokens_32(example, uncased):
    text = example[features.KEY_DOCUMENT_TEXT][:256]
    if uncased:
        text = text.lower()

    token_ids = _T5_TOKENIZER.encode(text, add_special_tokens=False, truncation=True, max_length=32)
    payload = ",".join(map(str, token_ids)).encode("utf-8")

    return hashlib.md5(payload).hexdigest()


def get_page_id(example):
    return pageids.url_to_page_id(example[features.KEY_DOCUMENT_URL])


def get_hash_for(example, value, uncased=False):
    if uncased:
        value = value.lower()
    return hashlib.md5(value.encode("utf-8")).hexdigest()


DEDUP_METHODS = {
    DEDUP_METHOD_FULL_TEXT_UNCASED: get_hash_for_full_text_uncased,
    DEDUP_METHOD_FULL_TEXT_CASED: get_hash_for_full_text_cased,
    DEDUP_METHOD_URL_UNCASED: get_hash_for_url_uncased,
    DEDUP_METHOD_URL_CASED: get_hash_for_url_cased,
    DEDUP_METHOD_TITLE_UNCASED: get_hash_for_title_uncased,
    DEDUP_METHOD_TITLE_CASED: get_hash_for_title_cased,
    DEDUP_METHOD_NCITITLE: get_hash_for_ncititle,
    DEDUP_METHOD_NCI_H1: get_hash_for_nci_h1,
    DEDUP_METHOD_H1_UNCASED: get_hash_for_h1_uncased,
    DEDUP_METHOD_H1_CASED: get_hash_for_h1_cased,
    DEDUP_METHOD_PAGEID: get_page_id,
    DEDUP_METHOD_TEXT4K: get_hash_for_text4k_uncased,
    DEDUP_METHOD_TEXT4K_CASED: get_hash_for_text4k_cased,
    DEDUP_METHOD_TOKENS_32: get_hash_for_tokens_32_uncased,
    DEDUP_METHOD_TOKENS_32_CASED: get_hash_for_tokens_32_cased,
}


def get_hash_method(method_name):
    if method_name == DEDUP_METHOD_NONE:
        return None

    if method_name not in DEDUP_METHODS:
        msg = ERR_UNSUPPORTED_DEDUP_METHOD.format(name=method_name)
        raise ValueError(msg)

    if method_name == DEDUP_METHOD_PAGEID:
        pageids.setup()

    if method_name == DEDUP_METHOD_NCITITLE or method_name == DEDUP_METHOD_NCI_H1:
        setup_tokenizer()

    if method_name == DEDUP_METHOD_TOKENS_32 or method_name == DEDUP_METHOD_TOKENS_32_CASED:
        setup_t5_tokenizer()

    return DEDUP_METHODS[method_name]
