import datasets
from tqdm.auto import tqdm

from redsi.formats import features as out_features

from . import deduplication, features, titles

DATASET_SPLIT_TRAIN = "train"
DATASET_SPLIT_VALIDATION = "validation"

TQDM_UNIT = "example"


def preprocess(ds, config, num_proc, cache_dir=None):
    column_names = ds.column_names
    columns_to_remove = [features.KEY_LONG_ANSWER_CANDIDATES, features.KEY_ANNOTATIONS]

    for split_name in column_names:
        split_column_names = column_names[split_name]
        split_columns_to_remove = []
        for column_name in columns_to_remove:
            if column_name in split_column_names:
                split_columns_to_remove.append(column_name)
        ds[split_name] = ds[split_name].remove_columns(split_columns_to_remove)

    hash_fn = deduplication.get_hash_method(config.dedup_method)

    if hash_fn is None:
        return preprocess_without_deduplication(ds, config, cache_dir=cache_dir)

    hash_keys = {}

    out_documents = []
    out_queries = {
        DATASET_SPLIT_TRAIN: [],
        DATASET_SPLIT_VALIDATION: [],
    }

    for split_name in ds:
        ds_split = ds[split_name]

        size = len(ds_split)
        desc = f"Preprocessing {split_name}"

        with tqdm(total=size, desc=desc, unit=TQDM_UNIT) as progress_bar:
            for example in ds_split:
                progress_bar.update(1)

                hash_key = -1 if hash_fn is None else hash_fn(example)
                if hash_key in hash_keys:
                    document_identifier = hash_keys[hash_key]
                    append_source_id(out_documents[int(document_identifier)], example)
                else:
                    document_identifier = str(len(out_documents))
                    if hash_key != -1:
                        hash_keys[hash_key] = document_identifier

                    document = make_document(document_identifier, example, config)
                    out_documents.append(document)

                query = make_query(document_identifier, example)
                out_queries[split_name].append(query)

    return datasets.DatasetDict(
        documents=datasets.Dataset.from_list(out_documents),
        train_queries=datasets.Dataset.from_list(out_queries[DATASET_SPLIT_TRAIN]),
        validation_queries=datasets.Dataset.from_list(out_queries[DATASET_SPLIT_VALIDATION]),
    )


def preprocess_without_deduplication(ds, config, cache_dir=None):
    split_offsets = {}
    offset = 0
    for split_name in [DATASET_SPLIT_TRAIN, DATASET_SPLIT_VALIDATION]:
        split_offsets[split_name] = offset
        offset += len(ds[split_name])

    return datasets.DatasetDict(
        documents=datasets.Dataset.from_generator(
            generate_documents_without_deduplication,
            gen_kwargs={"ds": ds, "config": config, "split_offsets": split_offsets},
            features=output_features(),
            cache_dir=cache_dir,
        ),
        train_queries=datasets.Dataset.from_generator(
            generate_queries_without_deduplication,
            gen_kwargs={
                "split": ds[DATASET_SPLIT_TRAIN],
                "split_offset": split_offsets[DATASET_SPLIT_TRAIN],
            },
            features=output_features(),
            cache_dir=cache_dir,
        ),
        validation_queries=datasets.Dataset.from_generator(
            generate_queries_without_deduplication,
            gen_kwargs={
                "split": ds[DATASET_SPLIT_VALIDATION],
                "split_offset": split_offsets[DATASET_SPLIT_VALIDATION],
            },
            features=output_features(),
            cache_dir=cache_dir,
        ),
    )


def generate_documents_without_deduplication(ds, config, split_offsets):
    for split_name in [DATASET_SPLIT_TRAIN, DATASET_SPLIT_VALIDATION]:
        split = ds[split_name]
        split_offset = split_offsets[split_name]
        for index, example in enumerate(tqdm(split, desc=f"Preprocessing {split_name}", unit=TQDM_UNIT)):
            yield make_document(str(split_offset + index), example, config)


def generate_queries_without_deduplication(split, split_offset):
    for index, example in enumerate(split):
        yield make_query(str(split_offset + index), example)


def output_features():
    return datasets.Features(
        {
            out_features.FEATURE_DOCID: datasets.Value("string"),
            out_features.FEATURE_KIND: datasets.Value("string"),
            out_features.FEATURE_TEXT: datasets.Value("string"),
            out_features.FEATURE_TITLE: datasets.Value("string"),
            out_features.FEATURE_SOURCE_IDS: datasets.Sequence(datasets.Value("string")),
        }
    )


def make_query(document_identifier, example):
    return {
        out_features.FEATURE_DOCID: document_identifier,
        out_features.FEATURE_KIND: out_features.FEATURE_KIND_QUERY,
        out_features.FEATURE_TEXT: example[features.KEY_QUESTION_TEXT],
        out_features.FEATURE_TITLE: "",
        out_features.FEATURE_SOURCE_IDS: [source_id(example)],
    }


def make_document(document_identifier, example, config):
    document_text = example[features.KEY_DOCUMENT_TEXT]

    if config.text_limit > 0:
        document_text = document_text[: config.text_limit]
    if config.lowercase:
        document_text = document_text.lower()

    title = titles.from_url(example[features.KEY_DOCUMENT_URL])

    return {
        out_features.FEATURE_DOCID: document_identifier,
        out_features.FEATURE_KIND: out_features.FEATURE_KIND_DOCUMENT,
        out_features.FEATURE_TEXT: document_text,
        out_features.FEATURE_TITLE: title,
        out_features.FEATURE_SOURCE_IDS: [source_id(example)],
    }


def source_id(example):
    return str(example[features.KEY_ID])


def append_source_id(document, example):
    current = document.setdefault(out_features.FEATURE_SOURCE_IDS, [])
    value = source_id(example)
    if value not in current:
        current.append(value)


stages = [
    preprocess,
]
