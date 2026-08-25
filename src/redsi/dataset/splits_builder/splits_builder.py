import datasets
from tqdm.auto import tqdm

from ...decorators.logging import log_function_start
from ...formats import formats

STR_SELECT_QUERIES = "[{name:<10}] select {size} queries"
STR_SELECT_DOCUMENTS = "[{name:<10}] select {size} documents"

# Errors
ERR_SELECT_QUERIES = "Could only select {selected_size}/{size} query rows."
ERR_SELECT_DOCUMENTS = "Could only select {selected_size}/{size} document rows."


@log_function_start("datasets.splits")
def splits(ds, config):
    ds_splits = []

    for split_config in config.datasets.splits:
        ds_split = build_split(ds, config, split_config)
        ds_splits.append((split_config.name, ds_split))

    return ds_splits


def build_split(ds, config, split_config):
    if split_config.train_queries == -1 and split_config.val_queries == -1:
        return ds

    documents = ds[formats.KEY_DOCUMENTS]
    queries = datasets.concatenate_datasets([ds[formats.KEY_TRAIN_QUERIES], ds[formats.KEY_VALIDATION_QUERIES]])
    queries = queries.shuffle(seed=config.arguments.seed)

    validation_docids, validation_queries = select_validation_queries(
        config.datasets.validation,
        queries,
        split_config.val_queries,
    )

    train_docids, train_queries = select_train_queries(
        config.datasets.train,
        queries,
        split_config.train_queries,
        split_config.unique_docs_in_train,
        validation_docids,
    )

    train_documents = select_documents_by_docids(
        config.datasets.train,
        documents,
        set(train_docids | validation_docids),
    )

    return datasets.DatasetDict(
        documents=train_documents,
        train_queries=train_queries,
        validation_queries=validation_queries,
    )


def select_validation_queries(name, queries, size):
    return select_queries(name, queries, size, True, None)


def select_train_queries(name, queries, size, enforce_unique_docids, skip_docids):
    return select_queries(name, queries, size, enforce_unique_docids, skip_docids)


def select_queries(
    name,
    queries,
    size,
    enforce_unique_docids,
    skip_docids=None,
):
    selected_docids = set()
    selected_indices = []

    desc = STR_SELECT_QUERIES.format(name=name, size=size)
    with tqdm(total=size, desc=desc) as progress_bar:
        for index, query in enumerate(queries):
            docid = query[formats.FEATURE_DOCID]

            if skip_docids is not None and docid in skip_docids:
                continue

            if enforce_unique_docids and docid in selected_docids:
                continue

            selected_docids.add(docid)
            selected_indices.append(index)

            progress_bar.update(1)
            if len(selected_indices) >= size:
                break

    if len(selected_indices) < size:
        msg = ERR_SELECT_QUERIES.format(selected_size=len(selected_indices), size=size)
        raise RuntimeError(msg)

    return selected_docids, queries.select(selected_indices)


def select_documents_by_docids(name, documents, docids):
    selected_indices = []

    size = len(docids)
    desc = STR_SELECT_DOCUMENTS.format(name=name, size=size)
    with tqdm(total=size, desc=desc) as progress_bar:
        for index, document in enumerate(documents):
            if document[formats.FEATURE_DOCID] in docids:
                selected_indices.append(index)

                progress_bar.update(1)
                if len(selected_indices) >= size:
                    break

    if len(selected_indices) < size:
        msg = ERR_SELECT_DOCUMENTS.format(selected_size=len(selected_indices), size=size)
        raise RuntimeError(msg)

    return documents.select(selected_indices)
