import gzip
import json
import os

import datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from . import features, text_style, text_utils, titles

# Flow 1: Load / Download
# =======================
#
# Input:
#   data_source = local folder
#   split
#   text_style = X
#
#
#                         ┌────────────────────────┐
#                         │  load split request    │
#                         └──────────┬─────────────┘
#                                    │
#                                    ▼
#                     try local styled simplified
#                     simplified-nq-{split}-X.jsonl.gz
#                                    │
#                           ┌────────┴────────┐
#                           │                 │
#                        exists             missing
#                           │                 │
#                           ▼                 ▼
#                      SIMPLIFIED[X]   try download styled simplified
#                                      from vingtetun/natural-questions
#                                             │
#                                    ┌────────┴────────┐
#                                    │                 │
#                                 success            fail
#                                    │                 │
#                                    ▼                 ▼
#                               SIMPLIFIED[X]   can use default simplified?
#                                                      │
#                                             ┌────────┴───────────────────────┐
#                                             │                                │
#                                            yes                               no
#                                             │                                │
#                                             ▼                                ▼
#                        try local default simplified                      try local full
#                        simplified-nq-{split}.jsonl.gz                   nq-{split}-all.jsonl.gz
#                                             │                                │
#                                    ┌────────┴────────┐                  ┌────┴────┐
#                                    │                 │                  │         │
#                                 exists           missing              exists   missing
#                                    │                 │                  │         │
#                                    ▼                 ▼                  ▼         │
#                         SIMPLIFIED[default]  try download default  load local full│
#                                              simplified                           │
#                                              from vingtetun                       │
#                                                   │                               │
#                                          ┌────────┴────────┐                      │
#                                          │                 │                      │
#                                       success            fail                     │
#                                          │                 │                      │
#                                          ▼                 ▼                      │
#                         SIMPLIFIED[default]         try local full                │
#                                                     nq-{split}-all                │
#                                                           │                       │
#                                                  ┌────────┴────────┐              │
#                                                  │                 │              │
#                                               exists             missing          │
#                                                  │                 │              │
#                                                  ▼                 ▼              ▼
#                                           load local full     load HF dataset full split
#                                                  │                 │              │
#                                                  │                 │              │
#                                                  └─────────────────┴──────────────┘
#                                                                    │
#                                                                    ▼
#                                                                   FULL
#
#
# Output:
#   raw split:
#     - SIMPLIFIED[X]
#     - SIMPLIFIED[default]
#     - FULL
#
#
# Flow 2: Processing / Normalization
# ==================================
#
# Input:
#   raw split:
#     - SIMPLIFIED[X]
#     - SIMPLIFIED[default]
#     - FULL
#   text_style = X
#
#
#                         ┌────────────────────────┐
#                         │      raw split         │
#                         └──────────┬─────────────┘
#                                    │
#                                    ▼
#                           cache exists for X?
#                                    │
#                           ┌────────┴────────┐
#                           │                 │
#                         yes                no
#                           │                 │
#                           ▼                 ▼
#                    SIMPLIFIED[X]      detect format
#                                             │
#                              ┌──────────────┴──────────────┐
#                              │                             │
#                        SIMPLIFIED                         FULL
#                              │                             │
#                              ▼                             ▼
#                   has document_text_style?       simplify_nq_example
#                              │                   with text_style = X
#                     ┌────────┴────────┐                    │
#                     │                 │                    │
#                    yes                no                   │
#                     │                 │                    │
#                     ▼                 ▼                    ▼
#             text_style == X?    assume default       SIMPLIFIED[X]
#                     │                 │
#             ┌───────┴───────┐         │
#             │               │         │
#            yes              no        │
#             │               │         │
#             ▼               ▼         ▼
#       SIMPLIFIED[X]       error   can derive X from default?
#                                      │
#                             ┌────────┴────────┐
#                             │                 │
#                            yes                no
#                             │                 │
#                             ▼                 ▼
#                    convert default -> X      error
#                             │
#                             ▼
#                       SIMPLIFIED[X]
#                             │
#                             ▼
#                    export jsonl.gz cache
#                             │
#                             ▼
#                       SIMPLIFIED[X]
#
#
# Output:
#   SIMPLIFIED[X]

###############################################################################
# Split names
###############################################################################

TRAIN_SPLIT_NAME = "train"
VALIDATION_SPLIT_NAME = "validation"

SPLIT_NAMES = [
    VALIDATION_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
]

###############################################################################
# Dataset formats
###############################################################################

DATASET_FORMAT_FULL = "full"
DATASET_FORMAT_HF_FULL = "hf_full"
DATASET_FORMAT_SIMPLE = "simple"

###############################################################################
# Text styles
###############################################################################

KEY_DOCUMENT_NORMALIZATION = "document_normalization"

###############################################################################
# Filenames
###############################################################################

SIMPLIFIED_TRAIN_FILENAME = "simplified-nq-train.jsonl.gz"
SIMPLIFIED_VALIDATION_FILENAME = "simplified-nq-dev.jsonl.gz"

FULL_TRAIN_FILENAME = "nq-train-all.jsonl.gz"
FULL_VALIDATION_FILENAME = "nq-dev-all.jsonl.gz"

SIMPLIFIED_TRAIN_FILENAME_STYLE = "simplified-nq-train-{text_style}.jsonl.gz"
SIMPLIFIED_VALIDATION_FILENAME_STYLE = "simplified-nq-dev-{text_style}.jsonl.gz"

CACHE_FILENAME = "simplified-nq-{split_name}-{text_style}.jsonl.gz"

###############################################################################
# Hugging Face
###############################################################################

HF_DATASET_SOURCE = "natural_questions"

HF_REPO_ID = "vingtetun/natural-questions"
HF_REPO_TYPE = "dataset"
HF_SUBDIR = "data"

###############################################################################
# Loading constants
###############################################################################

JSON_FORMAT = "json"
NPROC_SIMPLE_CONVERSION_DEFAULT = 8
NPROC_SIMPLE_CONVERSION_ENV = "NQ_CONVERSION_NUM_PROC"
SOURCE_CACHE_SUBDIR = ".cache/huggingface"

###############################################################################
# Errors
###############################################################################

ERR_UNKNOWN_DATASET_FORMAT = "Could not detect dataset format for split '{split_name}'."
ERR_TEXT_STYLE_MISMATCH = (
    "Split '{split_name}' has document_text_style={actual_styles}, but requested text_style='{expected_style}'."
)
ERR_FULL_ONLY_STYLE_FROM_SIMPLE = (
    "Requested text_style='{text_style}' requires full NQ format, but split '{split_name}' is already simplified."
)
ERR_UNSUPPORTED_SPLIT = "Unsupported split: '{split_name}'."
ERR_INVALID_NUM_PROC = "{name} must be a positive integer, got: {value}"

###############################################################################
# Public API
###############################################################################


def load(params, dataset_limit, cache_dir):
    data_dir = normalize_data_dir(params.data_source)
    source_cache_dir = resolve_source_cache_dir(data_dir)
    raw_splits = load_raw_splits(data_dir, params.text_style, source_cache_dir)
    return prepare_splits(raw_splits, params.text_style, params.dedup_method, source_cache_dir, data_dir)


###############################################################################
# Flow 1: Load / Download
###############################################################################


def load_raw_splits(data_dir, requested_text_style, cache_dir):
    _ensure_dir(data_dir)

    return {
        split_name: load_raw_split(data_dir, split_name, requested_text_style, cache_dir) for split_name in SPLIT_NAMES
    }


###############################################################################
# Flow 1: Load / Download
###############################################################################


def load_raw_split(data_dir, split_name, requested_text_style, cache_dir):
    simplified_path = try_resolve_simplified_path(
        data_dir,
        split_name,
        requested_text_style,
    )

    if simplified_path is not None:
        return load_jsonl_split(split_name, simplified_path, cache_dir)

    if text_style.requires_simplified_format(requested_text_style):
        default_path = try_resolve_default_simplified_path(data_dir, split_name)

        if default_path is not None:
            return load_jsonl_split(split_name, default_path, cache_dir)

    full_path = try_resolve_full_path(data_dir, split_name)

    if full_path is not None:
        return load_jsonl_split(split_name, full_path, cache_dir)

    return load_hf_full_split(split_name, cache_dir)


def try_resolve_simplified_path(data_dir, split_name, requested_text_style):
    filename = simplified_filename(split_name, requested_text_style)
    return try_resolve_local_or_hf_file(data_dir, filename)


def try_resolve_default_simplified_path(data_dir, split_name):
    filename = default_simplified_filename(split_name)
    return try_resolve_local_or_hf_file(data_dir, filename)


def try_resolve_full_path(data_dir, split_name):
    filename = full_filename(split_name)

    if _exists(data_dir, filename):
        return _local_path(data_dir, filename)

    return None


def try_resolve_local_or_hf_file(data_dir, filename):
    if _exists(data_dir, filename):
        return _local_path(data_dir, filename)

    return try_download_from_hf(filename, data_dir)


###############################################################################
# Flow 2: Processing / Normalization
###############################################################################


def prepare_splits(raw_splits, requested_text_style, dedup_method, cache_dir, data_dir):
    splits = {}

    for split_name, split in raw_splits.items():
        splits[split_name] = prepare_split(
            split_name,
            split,
            requested_text_style,
            dedup_method,
            cache_dir,
            data_dir,
        )

    return datasets.DatasetDict(splits)


def prepare_split(split_name, split, requested_text_style, dedup_method, cache_dir, data_dir):
    cached_path = _resolve_prepared_cache_path(data_dir, split_name, requested_text_style)

    if _valid_jsonl_gz_file(cached_path):
        cached = load_jsonl_split(split_name, cached_path, cache_dir)
        if is_usable_prepared_cache(cached, dedup_method):
            return cached

    prepared = prepare_uncached_split(split_name, split, requested_text_style)

    if should_cache_split(split, prepared, requested_text_style):
        export_split_jsonl_gz(prepared, cached_path)

    return prepared


def is_usable_prepared_cache(split, dedup_method):
    if not requires_ncititle_metadata(dedup_method):
        return True

    return features.KEY_NCI_DEDUP_TITLE in split.column_names


def requires_ncititle_metadata(dedup_method):
    return dedup_method == "ncititle"


def prepare_uncached_split(split_name, split, requested_text_style):
    dataset_format = detect_dataset_format(split)

    if dataset_format == DATASET_FORMAT_FULL:
        return convert_full_split(split_name, split, requested_text_style)

    if dataset_format == DATASET_FORMAT_HF_FULL:
        return convert_hf_full_split(split_name, split, requested_text_style)

    if dataset_format == DATASET_FORMAT_SIMPLE:
        return prepare_simple_split(split_name, split, requested_text_style)

    raise ValueError(ERR_UNKNOWN_DATASET_FORMAT.format(split_name=split_name))


def should_cache_split(raw_split, prepared_split, requested_text_style):
    raw_format = detect_dataset_format(raw_split)

    if raw_format in {DATASET_FORMAT_FULL, DATASET_FORMAT_HF_FULL}:
        return True

    if raw_format != DATASET_FORMAT_SIMPLE:
        return False

    if requested_text_style == text_style.TEXT_STYLE_SIMPLIFIED:
        return False

    return raw_split is not prepared_split or has_document_text_style(prepared_split)


###############################################################################
# Dataset format detection
###############################################################################


def detect_dataset_format(split):
    if has_document_text(split):
        return DATASET_FORMAT_SIMPLE

    if has_full_nq_columns(split):
        return DATASET_FORMAT_FULL

    if has_hf_full_nq_columns(split):
        return DATASET_FORMAT_HF_FULL

    return None


def has_hf_full_nq_columns(split):
    required_columns = {
        "id",
        "document",
        "question",
    }

    return required_columns.issubset(set(split.column_names))


def has_document_text(split):
    return features.KEY_DOCUMENT_TEXT in split.column_names


def has_document_text_style(split):
    return KEY_DOCUMENT_NORMALIZATION in split.column_names


def has_full_nq_columns(split):
    required_columns = {
        features.KEY_DOCUMENT_TITLE,
        features.KEY_DOCUMENT_HTML,
        features.KEY_DOCUMENT_TOKENS,
        features.KEY_QUESTION_TOKENS,
    }

    return required_columns.issubset(set(split.column_names))


###############################################################################
# Full -> simplified conversion
###############################################################################


def convert_full_split(split_name, split, requested_text_style):
    text_style.preload_text_style_resources(requested_text_style)
    convert_example = get_full_converter(split_name, requested_text_style)

    return split.map(
        convert_example,
        desc=f"Convert {split_name} full format to text_style={requested_text_style}",
        num_proc=get_conversion_num_proc(),
        remove_columns=full_columns_to_remove(split),
        fn_kwargs={
            "split_name": split_name,
            "requested_text_style": requested_text_style,
        },
        writer_batch_size=1000,
    )


def convert_hf_full_split(split_name, split, requested_text_style):
    text_style.preload_text_style_resources(requested_text_style)
    convert_example = get_hf_full_converter(split_name, requested_text_style)

    return split.map(
        convert_example,
        desc=f"Convert {split_name} HF full format to text_style={requested_text_style}",
        num_proc=get_conversion_num_proc(),
        remove_columns=list(split.column_names),
        fn_kwargs={
            "split_name": split_name,
            "requested_text_style": requested_text_style,
        },
        writer_batch_size=1000,
    )


def get_hf_full_converter(split_name, requested_text_style):
    if text_style.requires_full_format(requested_text_style):
        return convert_hf_full_example_with_full_style

    if text_style.requires_simplified_format(requested_text_style):
        return convert_hf_full_example_with_simplified_style

    raise ValueError(text_style.ERR_UNSUPPORTED_TEXT_STYLE.format(name=requested_text_style))


def convert_hf_full_example_with_full_style(example, split_name, requested_text_style):
    method = text_style.get_text_style_method_from_name(requested_text_style)
    original_like = hf_google_nq_to_original_row(example)
    simplified = text_utils.simplify_nq_example(original_like)
    add_title_metadata(simplified, original_like, split_name, features.KEY_ORIGINAL_FORMAT_FULL)

    simplified[features.KEY_DOCUMENT_TEXT] = text_style.apply_text_style(
        method,
        original_like,
        split_name=split_name,
    )
    simplified[KEY_DOCUMENT_NORMALIZATION] = requested_text_style

    return simplified


def convert_hf_full_example_with_simplified_style(example, split_name, requested_text_style):
    method = text_style.get_text_style_method_from_name(requested_text_style)
    original_like = hf_google_nq_to_original_row(example)
    simplified = text_utils.simplify_nq_example(original_like)
    add_title_metadata(simplified, original_like, split_name, features.KEY_ORIGINAL_FORMAT_FULL)
    style_row = copy_document_title_for_style(simplified, original_like)

    simplified[features.KEY_DOCUMENT_TEXT] = text_style.apply_text_style(
        method,
        style_row,
        split_name=split_name,
    )
    simplified[KEY_DOCUMENT_NORMALIZATION] = requested_text_style

    return simplified


def get_full_converter(split_name, requested_text_style):
    if text_style.requires_full_format(requested_text_style):
        return convert_full_example_with_full_style

    if text_style.requires_simplified_format(requested_text_style):
        return convert_full_example_with_simplified_style

    raise ValueError(text_style.ERR_UNSUPPORTED_TEXT_STYLE.format(name=requested_text_style))


def convert_full_example_with_full_style(example, split_name, requested_text_style):
    method = text_style.get_text_style_method_from_name(requested_text_style)
    simplified = text_utils.simplify_nq_example(example)
    add_title_metadata(simplified, example, split_name, features.KEY_ORIGINAL_FORMAT_FULL)

    simplified[features.KEY_DOCUMENT_TEXT] = text_style.apply_text_style(
        method,
        example,
        split_name=split_name,
    )
    simplified[KEY_DOCUMENT_NORMALIZATION] = requested_text_style

    return simplified


def convert_full_example_with_simplified_style(example, split_name, requested_text_style):
    method = text_style.get_text_style_method_from_name(requested_text_style)
    simplified = text_utils.simplify_nq_example(example)
    add_title_metadata(simplified, example, split_name, features.KEY_ORIGINAL_FORMAT_FULL)
    style_row = copy_document_title_for_style(simplified, example)

    simplified[features.KEY_DOCUMENT_TEXT] = text_style.apply_text_style(
        method,
        style_row,
        split_name=split_name,
    )
    simplified[KEY_DOCUMENT_NORMALIZATION] = requested_text_style

    return simplified


def copy_document_title_for_style(row, source):
    if features.KEY_DOCUMENT_TITLE not in source:
        return row

    row = dict(row)
    row[features.KEY_DOCUMENT_TITLE] = source[features.KEY_DOCUMENT_TITLE]
    return row


def add_title_metadata(row, source, split_name, original_format):
    row[features.KEY_ORIGINAL_FORMAT] = original_format

    url_title = get_url_title(row)
    h1_title = titles.from_h1(row[features.KEY_DOCUMENT_TEXT])
    nq_title = get_nq_title(source, url_title)

    row[features.KEY_DOCUMENT_URL_TITLE] = url_title
    row[features.KEY_DOCUMENT_H1_TITLE] = h1_title
    row[features.KEY_DOCUMENT_NQ_TITLE] = nq_title
    row[features.KEY_NCI_DEDUP_TITLE] = get_ncititle_dedup_title(split_name, h1_title, nq_title)


def get_url_title(row):
    try:
        return titles.from_url(row[features.KEY_DOCUMENT_URL])
    except (KeyError, IndexError, ValueError):
        return ""


def get_nq_title(source, url_title):
    return source.get(features.KEY_DOCUMENT_TITLE) or url_title


def get_ncititle_dedup_title(split_name, h1_title, nq_title):
    if split_name == TRAIN_SPLIT_NAME:
        return h1_title

    return nq_title


def full_columns_to_remove(split):
    candidate_columns = [
        features.KEY_DOCUMENT_TITLE,
        features.KEY_DOCUMENT_HTML,
        features.KEY_DOCUMENT_TOKENS,
        features.KEY_QUESTION_TOKENS,
    ]

    return [column for column in candidate_columns if column in split.column_names]


def hf_dict_of_lists_to_list_of_dicts(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if not isinstance(value, dict):
        return value

    keys = list(value.keys())
    if not keys:
        return []

    length = hf_common_list_length(value)
    if length is None:
        return value

    return [{key: hf_take_index(value[key], index) for key in keys} for index in range(length)]


def hf_common_list_length(value):
    for subvalue in value.values():
        if isinstance(subvalue, list):
            return len(subvalue)

        if isinstance(subvalue, dict):
            nested_length = hf_common_list_length(subvalue)
            if nested_length is not None:
                return nested_length

    return None


def hf_take_index(value, index):
    if isinstance(value, list):
        return normalize_hf_nested_value(value[index])

    if isinstance(value, dict):
        nested_length = hf_common_list_length(value)

        if nested_length is not None:
            return {key: hf_take_index(subvalue, index) for key, subvalue in value.items()}

        return {key: normalize_hf_nested_value(subvalue) for key, subvalue in value.items()}

    return value


def normalize_hf_nested_value(value):
    if isinstance(value, dict):
        converted = hf_dict_of_lists_to_list_of_dicts(value)
        return converted

    return value


def hf_google_nq_to_original_row(row):
    tokens = row["document"]["tokens"]

    document_tokens = [
        {
            "token": token,
            "html_token": is_html,
            "start_byte": start_byte,
            "end_byte": end_byte,
        }
        for token, is_html, start_byte, end_byte in zip(
            tokens["token"],
            tokens["is_html"],
            tokens["start_byte"],
            tokens["end_byte"],
        )
    ]

    long_answer_candidates = row.get("long_answer_candidates")
    if long_answer_candidates is None:
        long_answer_candidates = row["document"].get("long_answer_candidates", [])

    return {
        "example_id": row["id"],
        "document_url": row["document"]["url"],
        "document_html": row["document"]["html"],
        "document_tokens": document_tokens,
        "question_text": row["question"]["text"],
        "annotations": hf_dict_of_lists_to_list_of_dicts(row.get("annotations", [])),
        "long_answer_candidates": hf_dict_of_lists_to_list_of_dicts(long_answer_candidates),
    }


###############################################################################
# Simplified pipeline
###############################################################################


def prepare_simple_split(split_name, split, requested_text_style):
    if has_document_text_style(split):
        validate_document_text_style(split_name, split, requested_text_style)
        return stamp_simple_format(split_name, split, enrich_title_metadata=False)

    split = stamp_simple_format(split_name, split, enrich_title_metadata=True)

    if text_style.requires_simplified_format(requested_text_style):
        return convert_simple_split(split_name, split, requested_text_style)

    raise ValueError(
        ERR_FULL_ONLY_STYLE_FROM_SIMPLE.format(
            split_name=split_name,
            text_style=requested_text_style,
        )
    )


def stamp_simple_format(split_name, split, enrich_title_metadata):
    return split.map(
        stamp_simple_example,
        desc=f"Stamp {split_name} with simple format",
        num_proc=get_conversion_num_proc(),
        fn_kwargs={
            "split_name": split_name,
            "enrich_title_metadata": enrich_title_metadata,
        },
        writer_batch_size=1000,
    )


def stamp_simple_example(example, split_name, enrich_title_metadata):
    example = dict(example)

    if enrich_title_metadata and features.KEY_NCI_DEDUP_TITLE not in example:
        add_title_metadata(example, example, split_name, features.KEY_ORIGINAL_FORMAT_SIMPLE)
    else:
        example.setdefault(features.KEY_ORIGINAL_FORMAT, features.KEY_ORIGINAL_FORMAT_SIMPLE)

    return example


def validate_document_text_style(split_name, split, expected_style):
    actual_styles = set(split.unique(KEY_DOCUMENT_NORMALIZATION))

    if actual_styles != {expected_style}:
        raise ValueError(
            ERR_TEXT_STYLE_MISMATCH.format(
                split_name=split_name,
                actual_styles=sorted(actual_styles),
                expected_style=expected_style,
            )
        )


###############################################################################
# Simplified default -> styled conversion
###############################################################################


def convert_simple_split(split_name, split, requested_text_style):
    return split.map(
        convert_simple_example,
        desc=f"Convert simplified {split_name} default to text_style={requested_text_style}",
        num_proc=get_conversion_num_proc(),
        fn_kwargs={
            "split_name": split_name,
            "requested_text_style": requested_text_style,
        },
        writer_batch_size=1,
    )


def convert_simple_example(example, split_name, requested_text_style):
    example = dict(example)

    method = text_style.get_text_style_method_from_name(requested_text_style)
    example[features.KEY_DOCUMENT_TEXT] = text_style.apply_text_style(
        method,
        example,
        split_name=split_name,
    )
    example[KEY_DOCUMENT_NORMALIZATION] = requested_text_style

    return example


def get_conversion_num_proc():
    value = os.environ.get(NPROC_SIMPLE_CONVERSION_ENV)
    if value is None:
        return NPROC_SIMPLE_CONVERSION_DEFAULT

    try:
        num_proc = int(value)
    except ValueError as err:
        raise ValueError(ERR_INVALID_NUM_PROC.format(name=NPROC_SIMPLE_CONVERSION_ENV, value=value)) from err

    if num_proc < 1:
        raise ValueError(ERR_INVALID_NUM_PROC.format(name=NPROC_SIMPLE_CONVERSION_ENV, value=value))

    if num_proc == 1:
        return None

    return num_proc


###############################################################################
# Dataset loading helpers
###############################################################################


def load_jsonl_split(split_name, path, cache_dir):
    dataset = datasets.load_dataset(
        JSON_FORMAT,
        data_files={split_name: path},
        cache_dir=cache_dir,
    )

    return dataset[split_name]


def load_hf_full_split(split_name, cache_dir):
    return datasets.load_dataset(
        HF_DATASET_SOURCE,
        split=split_name,
        cache_dir=cache_dir,
    )


###############################################################################
# Filename helpers
###############################################################################


def simplified_filename(split_name, requested_text_style):
    if requested_text_style == text_style.TEXT_STYLE_SIMPLIFIED:
        return default_simplified_filename(split_name)

    return styled_simplified_filename(split_name, requested_text_style)


def styled_simplified_filename(split_name, requested_text_style):
    if split_name == TRAIN_SPLIT_NAME:
        return SIMPLIFIED_TRAIN_FILENAME_STYLE.format(text_style=requested_text_style)

    if split_name == VALIDATION_SPLIT_NAME:
        return SIMPLIFIED_VALIDATION_FILENAME_STYLE.format(text_style=requested_text_style)

    raise ValueError(ERR_UNSUPPORTED_SPLIT.format(split_name=split_name))


def default_simplified_filename(split_name):
    if split_name == TRAIN_SPLIT_NAME:
        return SIMPLIFIED_TRAIN_FILENAME

    if split_name == VALIDATION_SPLIT_NAME:
        return SIMPLIFIED_VALIDATION_FILENAME

    raise ValueError(ERR_UNSUPPORTED_SPLIT.format(split_name=split_name))


def full_filename(split_name):
    if split_name == TRAIN_SPLIT_NAME:
        return FULL_TRAIN_FILENAME

    if split_name == VALIDATION_SPLIT_NAME:
        return FULL_VALIDATION_FILENAME

    raise ValueError(ERR_UNSUPPORTED_SPLIT.format(split_name=split_name))


###############################################################################
# Local / HF file helpers
###############################################################################


def try_download_from_hf(filename, data_dir):
    try:
        return _download_from_hf(filename, data_dir)
    except Exception:
        return None


def _download_from_hf(filename, data_dir):
    path_in_repo = f"{HF_SUBDIR}/{filename}"

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=path_in_repo,
        local_dir=data_dir,
    )


def normalize_data_dir(data_dir):
    return os.path.join(data_dir, HF_SUBDIR)


def resolve_source_cache_dir(data_dir):
    return os.path.join(data_dir, SOURCE_CACHE_SUBDIR)


def _local_path(data_dir, filename):
    return os.path.join(data_dir, filename)


def _exists(data_dir, filename):
    return _valid_file(_local_path(data_dir, filename))


def _valid_file(path):
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def _valid_jsonl_gz_file(path):
    if not _valid_file(path):
        return False

    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


###############################################################################
# Cache helpers
###############################################################################


def _resolve_prepared_cache_path(data_dir, split_name, requested_text_style):
    if data_dir is None:
        return None

    filename = simplified_filename(split_name, requested_text_style)
    return os.path.join(data_dir, filename)


def export_split_jsonl_gz(split, path):
    if path is None:
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = f"{path}.tmp"

    with gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=1) as out:
        for row in tqdm(split, desc=f"Writing {os.path.basename(path)}"):
            out.write(json.dumps(row, ensure_ascii=False))
            out.write("\n")

    os.replace(tmp_path, path)

    return path
