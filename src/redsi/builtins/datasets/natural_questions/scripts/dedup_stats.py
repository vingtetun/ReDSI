#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

import datasets
from prettytable import PrettyTable, TableStyle
from tqdm.auto import tqdm

from redsi.builtins.datasets.natural_questions import features, text_style
from redsi.builtins.datasets.natural_questions.dataset import make_query
from redsi.builtins.datasets.natural_questions.deduplication import (
    KEY_FULL_TEXT_CASED_MD5,
    KEY_FULL_TEXT_UNCASED_MD5,
    get_hash_for,
    get_hash_method,
)
from redsi.builtins.datasets.natural_questions.loader import load
from redsi.formats import features as out_features

DATASET_SPLIT_TRAIN = "train"
DATASET_SPLIT_VALIDATION = "validation"
DEFAULT_SPLITS = [DATASET_SPLIT_TRAIN, DATASET_SPLIT_VALIDATION]

KEY_TRAIN_DOCUMENTS = "train_documents"
KEY_VALIDATION_DOCUMENTS = "validation_documents"
KEY_UNION_DOCUMENTS = "union"

KEY_HASH = "hash"


def preprocess(ds, config, splits=DEFAULT_SPLITS):
    ds = ds.remove_columns([features.KEY_ID, features.KEY_LONG_ANSWER_CANDIDATES, features.KEY_ANNOTATIONS])

    out_documents = {}
    out_queries = {}

    def count_documents():
        total = 0
        for key in out_documents:
            total += len(out_documents[key])
        return total

    def get_documents(split_name):
        return as_dataset_dict(out_documents.get(split_name, []))

    def get_queries(split_name):
        return as_dataset_dict(out_queries.get(split_name, []))

    def as_dataset_dict(items):
        return datasets.Dataset.from_list(items)

    for split_name in splits:
        out_documents[split_name] = []
        out_queries[split_name] = []
        ds_split = ds[split_name]

        size = len(ds_split)
        desc = f"Preprocessing {split_name}"

        with tqdm(total=size, desc=desc) as progress_bar:
            for example in ds_split:
                progress_bar.update(1)

                document_identifier = count_documents()
                document = make_document(document_identifier, example, config)
                out_documents[split_name].append(document)

                query = make_query(document_identifier, example)
                out_queries[split_name].append(query)

    return datasets.DatasetDict(
        train_documents=get_documents(DATASET_SPLIT_TRAIN),
        validation_documents=get_documents(DATASET_SPLIT_VALIDATION),
        train_queries=get_queries(DATASET_SPLIT_TRAIN),
        validation_queries=get_queries(DATASET_SPLIT_VALIDATION),
    )


def make_document(document_identifier, example, config):
    text_style_fn = text_style.get_text_style_method(config)
    document_text = text_style_fn(example[features.KEY_DOCUMENT_TEXT])

    document_text_cased_md5 = get_hash_for(example, document_text, False)
    document_text_uncased_md5 = get_hash_for(example, document_text, True)
    if config.text_limit > 0:
        document_text = document_text[: config.text_limit]
    if config.lowercase:
        document_text = document_text.lower()

    return {
        out_features.FEATURE_DOCID: document_identifier,
        features.KEY_DOCUMENT_URL: example[features.KEY_DOCUMENT_URL],
        out_features.FEATURE_KIND: out_features.FEATURE_KIND_DOCUMENT,
        features.KEY_DOCUMENT_TEXT: document_text,
        KEY_FULL_TEXT_CASED_MD5: document_text_cased_md5,
        KEY_FULL_TEXT_UNCASED_MD5: document_text_uncased_md5,
        features.KEY_ORIGINAL_FORMAT: example[features.KEY_ORIGINAL_FORMAT],
    }


def uniques(ds, method_name):
    hash_fn = get_hash_method(method_name)

    def hash_wrapper(example):
        return {KEY_HASH: hash_fn(example)}

    ds = ds.map(hash_wrapper, desc=method_name, keep_in_memory=True)
    return set(ds.unique(KEY_HASH))


def unique_documents(ds, method_name):
    train = uniques(ds[KEY_TRAIN_DOCUMENTS], method_name)
    validation = uniques(ds[KEY_VALIDATION_DOCUMENTS], method_name)
    union = set(train | validation)

    return len(train), len(validation), len(union)


def build_table(ds):
    train_count = len(ds[KEY_TRAIN_DOCUMENTS])
    validation_count = len(ds[KEY_VALIDATION_DOCUMENTS])
    union_count = train_count + validation_count

    train_counts = [train_count]
    validation_counts = [validation_count]
    union_counts = [union_count]

    method_names = [
        "full-text-cased",
        "full-text",
        "text4k-cased",
        "text4k",
        "url",
        "title-cased",
        "title",
        "h1-cased",
        "h1",
        "ncititle",
        "nci-h1",
        "pageid",
        "tokens-32",
        "tokens-32-cased",
    ]
    for method_name in method_names:
        train_count, validation_count, union_count = unique_documents(ds, method_name)
        train_counts.append(train_count)
        validation_counts.append(validation_count)
        union_counts.append(union_count)

    train_row = ["Train"]
    validation_row = ["Validation"]
    union_row = ["Union"]

    train_row.extend(train_counts)
    validation_row.extend(validation_counts)
    union_row.extend(union_counts)

    headers = [
        "",
        "# docs",
        "full-text",
        "lower(full-text)",
        "text4k",
        "lower(text4k) (DSI)",
        "url",
        "title",
        "lower(title)",
        "h1",
        "lower(h1)",
        "ncititle",
        "nci-h1",
        "pageid",
        "tokens-32",
        "tokens-32-cased",
    ]

    table = PrettyTable()
    table.field_names = headers
    table.add_row(train_row)
    table.add_row(validation_row)
    table.add_divider()
    table.add_row(union_row)

    table.align = "r"
    table.align[""] = "l"
    table.set_style(TableStyle.SINGLE_BORDER)

    return table


@dataclass
class Params:
    data_source: str
    text_style: str = "content"
    text_limit: int = 4000
    dedup_method: str = "none"
    lowercase: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="The source folder that contains the dataset train/validation files")
    args = parser.parse_args()

    params = Params(args.source)
    ds = load(params, -1, None)
    ds = preprocess(ds, params)

    table = build_table(ds)
    print(table)


if __name__ == "__main__":
    main()
