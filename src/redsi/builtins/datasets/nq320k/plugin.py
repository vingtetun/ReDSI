from redsi.plugins import Param

from .package import load_package

PLUGIN_DESCRIPTION = "Load packaged NQ320K document/query artifacts from local disk or Hugging Face."
PLUGIN_PARAMS = [
    Param(
        "data_source",
        "str",
        required=True,
        help="Local package directory, local repository root, or Hugging Face repository id.",
    ),
    Param(
        "config_name",
        "str",
        default="",
        help="Package config such as 'htmlnorm-pageid' or a relative package path such as 'data/htmlnorm/pageid'.",
    ),
    Param(
        "revision",
        "str",
        default="",
        help="Optional Hugging Face revision.",
    ),
    Param(
        "repo_type",
        "str",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repository type when data_source is a repository id.",
    ),
    Param(
        "text_limit",
        "int",
        default=-1,
        help="Maximum document text length. Use <= 0 to keep full text.",
    ),
    Param(
        "lowercase",
        "bool",
        default=False,
        help="Lowercase document text after loading.",
    ),
]


class NQ320KDatasetPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def load(self, params, dataset_limit, cache_dir):
        ds = load_package(
            params.data_source,
            config_name=params.config_name or None,
            revision=params.revision or None,
            repo_type=params.repo_type,
            cache_dir=cache_dir,
            text_limit=params.text_limit,
            lowercase=params.lowercase,
        )
        return apply_dataset_limit(ds, dataset_limit)

    def run(self):
        return [passthrough]


def apply_dataset_limit(ds, dataset_limit):
    if dataset_limit <= 0:
        return ds

    train_queries = ds["train_queries"].select(range(min(dataset_limit, len(ds["train_queries"]))))
    validation_queries = ds["validation_queries"].select(range(min(dataset_limit, len(ds["validation_queries"]))))
    keep_docids = set(train_queries["docid"]) | set(validation_queries["docid"])

    def keep_document(row):
        return row["docid"] in keep_docids

    documents = ds["documents"].filter(keep_document, desc="Select nq320k limited documents")
    return ds.__class__(
        documents=documents,
        train_queries=train_queries,
        validation_queries=validation_queries,
    )


def passthrough(ds, params, num_proc):
    return ds
