import datasets

from redsi.builtins.datasets.msmarco_documents import features
from redsi.formats import features as out_features

###############################################################################
# Split names
###############################################################################

DOCUMENTS_SPLIT_NAME = "documents"
TRAIN_QUERIES_SPLIT_NAME = "train_queries"
VALIDATION_QUERIES_SPLIT_NAME = "validation_queries"

QUERY_SPLIT_NAMES = {
    TRAIN_QUERIES_SPLIT_NAME,
    VALIDATION_QUERIES_SPLIT_NAME,
}

TEXT_STYLE_CONTENT = "content"
TEXT_STYLE_TITLE_CONTENT = "title-content"

ERR_UNSUPPORTED_TEXT_STYLE = (
    "Unsupported MS MARCO document text_style={text_style!r}; expected one of: "
    f"{TEXT_STYLE_CONTENT}, {TEXT_STYLE_TITLE_CONTENT}"
)


###############################################################################
# Helpers
###############################################################################


def maybe_lowercase(text, config):
    text = text or ""

    if getattr(config, "lowercase", False):
        return text.lower()

    return text


def maybe_limit_text(text, config):
    text = text or ""

    text_limit = getattr(config, "text_limit", None)
    if text_limit is None:
        return text

    if text_limit <= 0:
        return text

    return text[:text_limit]


def normalize_text(text, config):
    text = text or ""
    text = maybe_limit_text(text, config)
    text = maybe_lowercase(text, config)
    return text


def join_non_empty(parts, separator=" "):
    return separator.join(part for part in parts if part)


def format_document_text(title, body, config):
    text_style = getattr(config, "text_style", TEXT_STYLE_CONTENT)

    if text_style == TEXT_STYLE_CONTENT:
        return normalize_text(body, config)

    if text_style == TEXT_STYLE_TITLE_CONTENT:
        title = maybe_lowercase(title, config).strip()
        body = normalize_text(body, config).strip()
        return join_non_empty([title, body])

    raise ValueError(ERR_UNSUPPORTED_TEXT_STYLE.format(text_style=text_style))


###############################################################################
# Row conversion
###############################################################################


def make_query(example, config):
    query_text = maybe_lowercase(example[features.KEY_QUESTION_TEXT], config)

    return {
        out_features.FEATURE_DOCID: example[features.KEY_ID],
        out_features.FEATURE_KIND: out_features.FEATURE_KIND_QUERY,
        out_features.FEATURE_TEXT: query_text,
        out_features.FEATURE_TITLE: "",
    }


def make_document(example, config):
    title = example.get(features.KEY_DOCUMENT_TITLE, "") or ""
    body = example.get(features.KEY_DOCUMENT_TEXT, "") or ""

    title = maybe_lowercase(title, config)
    text = format_document_text(title, body, config)

    return {
        out_features.FEATURE_DOCID: example[features.KEY_ID],
        out_features.FEATURE_KIND: out_features.FEATURE_KIND_DOCUMENT,
        out_features.FEATURE_TEXT: text,
        out_features.FEATURE_TITLE: title,
    }


###############################################################################
# Split preprocessing
###############################################################################


def preprocess_documents(split, config, num_proc):
    return split.map(
        lambda example: make_document(example, config),
        remove_columns=split.column_names,
        num_proc=num_proc,
        desc="Preprocessing documents",
    )


def preprocess_queries(split, config, num_proc, desc):
    return split.map(
        lambda example: make_query(example, config),
        remove_columns=split.column_names,
        num_proc=num_proc,
        desc=desc,
    )


def preprocess(ds, config, num_proc):
    out = {}

    if DOCUMENTS_SPLIT_NAME in ds:
        out[DOCUMENTS_SPLIT_NAME] = preprocess_documents(
            ds[DOCUMENTS_SPLIT_NAME],
            config,
            num_proc,
        )

    if TRAIN_QUERIES_SPLIT_NAME in ds:
        out[TRAIN_QUERIES_SPLIT_NAME] = preprocess_queries(
            ds[TRAIN_QUERIES_SPLIT_NAME],
            config,
            num_proc,
            desc="Preprocessing train queries",
        )

    if VALIDATION_QUERIES_SPLIT_NAME in ds:
        out[VALIDATION_QUERIES_SPLIT_NAME] = preprocess_queries(
            ds[VALIDATION_QUERIES_SPLIT_NAME],
            config,
            num_proc,
            desc="Preprocessing validation queries",
        )

    return datasets.DatasetDict(out)


###############################################################################
# Pipeline
###############################################################################

stages = [
    preprocess,
]
