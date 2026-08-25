import gzip
import json
import os
from pathlib import Path

import datasets

from redsi.formats import formats

FORMAT_NAME = "nq320k_package"
FORMAT_VERSION = 1
METADATA_FILENAME = "metadata.json"

SPLIT_DOCUMENTS = formats.KEY_DOCUMENTS
SPLIT_TRAIN_QUERIES = formats.KEY_TRAIN_QUERIES
SPLIT_VALIDATION_QUERIES = formats.KEY_VALIDATION_QUERIES
PACKAGE_SPLITS = [
    SPLIT_DOCUMENTS,
    SPLIT_TRAIN_QUERIES,
    SPLIT_VALIDATION_QUERIES,
]

DATA_FILE_BY_SPLIT = {
    SPLIT_DOCUMENTS: "documents.jsonl.gz",
    SPLIT_TRAIN_QUERIES: "train_queries.jsonl.gz",
    SPLIT_VALIDATION_QUERIES: "validation_queries.jsonl.gz",
}

ERR_MISSING_SPLIT = "nq320k package: missing split '{split_name}'"
ERR_INVALID_SPLIT = "nq320k package: split '{split_name}' must be a Hugging Face Dataset, got {split_type}"
ERR_MISSING_FILE = "nq320k package: missing required file '{path}'"
ERR_MISSING_DOCID = "nq320k package: split '{split_name}' references missing docids"
ERR_NO_PACKAGE_PATH = (
    "nq320k package: could not resolve package path for data_source={data_source!r}, config_name={config_name!r}"
)
ERR_HF_CONFIG_REQUIRED = "nq320k package: config_name is required when loading from Hugging Face"
ERR_UNSUPPORTED_CONFIG_NAME = "nq320k package: unsupported config_name '{config_name}'"


def write_package(ds, out_dir, *, metadata=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    validate_package_dataset(ds)

    for split_name in PACKAGE_SPLITS:
        write_jsonl_gz(ds[split_name], out_path / DATA_FILE_BY_SPLIT[split_name])

    package_metadata = build_metadata(ds, metadata=metadata)
    metadata_path = out_path / METADATA_FILENAME
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(package_metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(tmp_path, metadata_path)


def load_package(
    data_source,
    *,
    config_name=None,
    revision=None,
    repo_type="dataset",
    cache_dir=None,
    text_limit=-1,
    lowercase=False,
):
    package_path = resolve_package_path(
        data_source=data_source,
        config_name=config_name,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
    )
    ensure_package_files(package_path)

    data_files = {split_name: str(package_path / filename) for split_name, filename in DATA_FILE_BY_SPLIT.items()}
    ds = datasets.DatasetDict(
        {
            split_name: datasets.load_dataset(
                "json",
                data_files={split_name: data_file},
                split=split_name,
                cache_dir=cache_dir,
            )
            for split_name, data_file in data_files.items()
        }
    )

    ds = preprocess_loaded_package(ds, text_limit=text_limit, lowercase=lowercase)
    validate_package_dataset(ds)
    validate_query_docids(ds)
    return ds


def resolve_package_path(*, data_source, config_name=None, revision=None, repo_type="dataset", cache_dir=None):
    source_path = Path(data_source).expanduser()
    if source_path.exists():
        return resolve_local_package_path(source_path, config_name)

    if not config_name:
        raise ValueError(ERR_HF_CONFIG_REQUIRED)

    from huggingface_hub import snapshot_download

    config_path = config_name_to_path(config_name)
    snapshot_path = Path(
        snapshot_download(
            repo_id=data_source,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                f"{config_path}/*",
                f"{config_path}/**",
            ],
        )
    )
    return resolve_local_package_path(snapshot_path, config_name)


def resolve_local_package_path(source_path, config_name=None):
    if is_package_dir(source_path):
        return source_path

    if config_name:
        candidate = source_path / config_name_to_path(config_name)
        if is_package_dir(candidate):
            return candidate

    raise FileNotFoundError(ERR_NO_PACKAGE_PATH.format(data_source=str(source_path), config_name=config_name))


def config_name_to_path(config_name):
    if "/" in config_name:
        return Path(config_name)

    normalized = config_name.replace("_", "-")
    for grouping in sorted(grouping_names(), key=len, reverse=True):
        suffix = f"-{grouping}"
        if normalized.endswith(suffix):
            style = normalized[: -len(suffix)]
            if style not in style_names():
                raise ValueError(ERR_UNSUPPORTED_CONFIG_NAME.format(config_name=config_name))
            return Path("data") / style / grouping

    if normalized not in style_names():
        raise ValueError(ERR_UNSUPPORTED_CONFIG_NAME.format(config_name=config_name))

    return Path("data") / normalized


def grouping_names():
    return {"none", "pageid", "ncititle", "url", "text4k"}


def style_names():
    return {"simplified", "simplenorm", "ncinorm", "htmlnorm"}


def is_package_dir(path):
    return (path / METADATA_FILENAME).exists() and all(
        (path / filename).exists() for filename in DATA_FILE_BY_SPLIT.values()
    )


def ensure_package_files(path):
    for filename in [METADATA_FILENAME, *DATA_FILE_BY_SPLIT.values()]:
        filepath = path / filename
        if not filepath.exists():
            raise FileNotFoundError(ERR_MISSING_FILE.format(path=filepath))


def preprocess_loaded_package(ds, *, text_limit, lowercase):
    if text_limit <= 0 and not lowercase:
        return ds

    def preprocess_document(batch):
        texts = batch[formats.FEATURE_TEXT]
        out = []
        for text in texts:
            if text_limit > 0:
                text = text[:text_limit]
            if lowercase:
                text = text.lower()
            out.append(text)
        return {formats.FEATURE_TEXT: out}

    ds = datasets.DatasetDict(dict(ds))
    ds[SPLIT_DOCUMENTS] = ds[SPLIT_DOCUMENTS].map(preprocess_document, batched=True, desc="Preprocess nq320k documents")
    return ds


def validate_package_dataset(ds):
    for split_name in PACKAGE_SPLITS:
        if split_name not in ds:
            raise ValueError(ERR_MISSING_SPLIT.format(split_name=split_name))
        if not isinstance(ds[split_name], datasets.Dataset):
            raise TypeError(
                ERR_INVALID_SPLIT.format(
                    split_name=split_name,
                    split_type=type(ds[split_name]),
                )
            )


def validate_query_docids(ds):
    docids = set(str(docid) for docid in ds[SPLIT_DOCUMENTS][formats.FEATURE_DOCID])
    for split_name in [SPLIT_TRAIN_QUERIES, SPLIT_VALIDATION_QUERIES]:
        missing = False
        for docid in ds[split_name][formats.FEATURE_DOCID]:
            if str(docid) not in docids:
                missing = True
                break
        if missing:
            raise ValueError(ERR_MISSING_DOCID.format(split_name=split_name))


def build_metadata(ds, *, metadata=None):
    metadata = dict(metadata or {})
    metadata.update(
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "splits": {
                split_name: {
                    "num_rows": len(ds[split_name]),
                    "columns": list(ds[split_name].column_names),
                    "features": {name: str(feature) for name, feature in ds[split_name].features.items()},
                    "data_file": DATA_FILE_BY_SPLIT[split_name],
                }
                for split_name in PACKAGE_SPLITS
            },
        }
    )
    return metadata


def write_jsonl_gz(ds, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as file:
        for row in ds:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")
    os.replace(tmp_path, path)
