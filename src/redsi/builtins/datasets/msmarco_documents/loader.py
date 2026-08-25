import os

import datasets

###############################################################################
# Data
###############################################################################

DATA_SUBDIR = "data"

###############################################################################
# Split names
###############################################################################

TRAIN_QUERIES_SPLIT_NAME = "train_queries"
VALIDATION_QUERIES_SPLIT_NAME = "validation_queries"
DOCUMENTS_SPLIT_NAME = "documents"

SPLIT_NAMES = [
    TRAIN_QUERIES_SPLIT_NAME,
    VALIDATION_QUERIES_SPLIT_NAME,
    DOCUMENTS_SPLIT_NAME,
]

###############################################################################
# Loading
###############################################################################

JSON_FORMAT = "json"

###############################################################################
# Filenames
###############################################################################

DOCUMENTS_FILENAME = "documents.jsonl.gz"
TRAIN_FILENAME = "train_queries.jsonl.gz"
VALIDATION_FILENAME = "validation_queries.jsonl.gz"

###############################################################################
# Errors
###############################################################################

ERR_UNSUPPORTED_SPLIT = "Unsupported split: '{split_name}'."

###############################################################################
# Public API
###############################################################################


def load(params, dataset_limit, cache_dir):
    data_dir = normalize_data_dir(params.data_source)

    raw_splits = load_raw_splits(
        data_dir=data_dir,
        cache_dir=cache_dir,
    )

    return prepare_splits(
        raw_splits=raw_splits,
        cache_dir=cache_dir,
    )


###############################################################################
# Paths
###############################################################################


def normalize_data_dir(data_dir):
    return os.path.join(data_dir, DATA_SUBDIR)


###############################################################################
# Dataset loading
###############################################################################


def load_raw_splits(data_dir, cache_dir):
    _ensure_dir(data_dir)

    return {
        split_name: load_raw_split(
            data_dir=data_dir,
            split_name=split_name,
            cache_dir=cache_dir,
        )
        for split_name in SPLIT_NAMES
    }


def load_raw_split(data_dir, split_name, cache_dir):
    filename = default_filename(split_name)
    filepath = os.path.join(data_dir, filename)

    return load_jsonl_split(
        split_name=split_name,
        path=filepath,
        cache_dir=cache_dir,
    )


def load_jsonl_split(split_name, path, cache_dir):
    dataset = datasets.load_dataset(
        JSON_FORMAT,
        data_files={split_name: path},
        cache_dir=cache_dir,
    )

    return dataset[split_name]


###############################################################################
# Filenames
###############################################################################


def default_filename(split_name):
    if split_name == TRAIN_QUERIES_SPLIT_NAME:
        return TRAIN_FILENAME

    if split_name == VALIDATION_QUERIES_SPLIT_NAME:
        return VALIDATION_FILENAME

    if split_name == DOCUMENTS_SPLIT_NAME:
        return DOCUMENTS_FILENAME

    raise ValueError(
        ERR_UNSUPPORTED_SPLIT.format(
            split_name=split_name,
        )
    )


###############################################################################
# Utils
###############################################################################


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


###############################################################################
# Processing
###############################################################################


def prepare_splits(raw_splits, cache_dir):
    splits = {}

    for split_name, split in raw_splits.items():
        splits[split_name] = prepare_split(
            split_name=split_name,
            split=split,
            cache_dir=cache_dir,
        )

    return datasets.DatasetDict(splits)


def prepare_split(split_name, split, cache_dir):
    return split
