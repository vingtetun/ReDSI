import os

import datasets

from ...plugins import Param

JSONL_FORMAT = "jsonl"
PARQUET_FORMAT = "parquet"
HF_DISK_FORMAT = "hf_disk"

PLUGIN_DESCRIPTION = "Write dataset splits to disk. Supports jsonl, parquet, and HuggingFace save_to_disk."

PLUGIN_PARAMS = [
    Param(
        "format",
        "str",
        default=JSONL_FORMAT,
        choices=[JSONL_FORMAT, PARQUET_FORMAT, HF_DISK_FORMAT],
        help="Export format.",
    ),
]

ERR_UNSUPPORTED_FORMAT = "export writer: unsupported format '{export_format}'"
ERR_INVALID_DATASET = "export writer: split '{split_name}' must be a HuggingFace Dataset, got {dataset_type}"


class DiskWriterPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        export_config = _build_export_config(
            ds=ds,
            out_dir=out_dir,
            export_format=params.format,
        )

        export(params.format, export_config)


def _build_export_config(ds, out_dir, export_format):
    extension = "" if export_format == HF_DISK_FORMAT else f".{export_format}"
    export_config = {}

    for split_name, split_ds in ds.items():
        _validate_dataset(split_name, split_ds)

        filepath = os.path.join(out_dir, f"{split_name}{extension}")
        export_config[filepath] = split_ds

    return export_config


def _validate_dataset(split_name, split_ds):
    if not isinstance(split_ds, datasets.Dataset):
        raise ValueError(
            ERR_INVALID_DATASET.format(
                split_name=split_name,
                dataset_type=type(split_ds),
            )
        )


def export(export_format, export_config):
    export_format = export_format.lower()

    if export_format == JSONL_FORMAT:
        export_to_jsonl(export_config)
        return

    if export_format == PARQUET_FORMAT:
        export_to_parquet(export_config)
        return

    if export_format == HF_DISK_FORMAT:
        export_to_hf_disk(export_config)
        return

    raise ValueError(ERR_UNSUPPORTED_FORMAT.format(export_format=export_format))


def export_to_jsonl(export_config):
    for filepath, ds in export_config.items():
        ds.to_json(filepath)


def export_to_parquet(export_config):
    for filepath, ds in export_config.items():
        ds.to_parquet(filepath)


def export_to_hf_disk(export_config):
    for filepath, ds in export_config.items():
        ds.save_to_disk(filepath)
