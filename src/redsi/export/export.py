import json
import os

from omegaconf import OmegaConf

from ..decorators.logging import log_function_start

DEFAULT_SPLIT_NAME = "default"
METADATA_FILENAME = "metadata.json"
METADATA_VERSION = 1
HF_DISK_FORMAT = "hf_disk"


@log_function_start("plugin")
def export(run_config, plugin, plugin_params, data):
    if isinstance(data, list):
        for ds_split_name, ds_split in data:
            _export(run_config, plugin, plugin_params, ds_split, ds_split_name)
    else:
        _export(run_config, plugin, plugin_params, data, DEFAULT_SPLIT_NAME)


def _export(run_config, plugin, plugin_params, ds_split, ds_split_name):
    base_out_dir = run_config.arguments.out_dir
    split_out_dir = os.path.join(base_out_dir, ds_split_name)
    print(f"Writing {split_out_dir}")
    os.makedirs(split_out_dir, exist_ok=True)
    plugin.run(ds_split, plugin_params, split_out_dir)
    write_metadata(run_config, plugin, plugin_params, ds_split, ds_split_name, split_out_dir)


def write_metadata(run_config, plugin, plugin_params, ds, ds_split_name, out_dir):
    metadata_path = os.path.join(out_dir, METADATA_FILENAME)
    metadata = build_metadata(run_config, plugin, plugin_params, ds, ds_split_name, out_dir)

    tmp_path = f"{metadata_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2, sort_keys=True)
        fout.write("\n")

    os.replace(tmp_path, metadata_path)


def build_metadata(run_config, plugin, plugin_params, ds, ds_split_name, out_dir):
    export_params = _to_jsonable(plugin_params)
    export_format = export_params.get("format")

    return {
        "metadata_version": METADATA_VERSION,
        "dataset_split": ds_split_name,
        "out_dir": out_dir,
        "export": {
            "plugin": _plugin_identity(plugin),
            "params": export_params,
            "format": export_format,
        },
        "config": _build_config_summary(run_config),
        "splits": _build_split_metadata(ds, export_format),
        "token_table": _build_token_table_metadata(ds, export_format),
    }


def _build_config_summary(run_config):
    cfg = OmegaConf.to_container(run_config, resolve=True)
    return {
        "datasets": cfg.get("datasets"),
        "identifiers": cfg.get("identifiers"),
        "augmentation": cfg.get("augmentation"),
        "transformation": cfg.get("transformation"),
        "arguments": cfg.get("arguments"),
    }


def _build_split_metadata(ds, export_format):
    out = {}

    for split_name, split_ds in ds.items():
        out[split_name] = {
            "num_rows": len(split_ds),
            "columns": list(split_ds.column_names),
            "features": _features_to_jsonable(split_ds),
            "data_file": _exported_split_filename(split_name, export_format),
        }

    return out


def _build_token_table_metadata(ds, export_format):
    if "tokens" not in ds:
        return None

    tokens_ds = ds["tokens"]
    columns = set(tokens_ds.column_names)
    metadata = {
        "split": "tokens",
        "num_rows": len(tokens_ds),
        "columns": list(tokens_ds.column_names),
        "data_file": _exported_split_filename("tokens", export_format),
    }

    if {"base_vocab_size", "tokens"}.issubset(columns) and len(tokens_ds) > 0:
        row = tokens_ds[0]
        tokens = row.get("tokens") or []
        base_vocab_size = int(row["base_vocab_size"])
        metadata.update(
            {
                "kind": "compact_added_tokens",
                "base_vocab_size": base_vocab_size,
                "num_tokens": len(tokens),
                "vocab_size": base_vocab_size + len(tokens),
            }
        )
    elif "token" in columns:
        metadata.update(
            {
                "kind": "token_rows",
                "num_tokens": len(tokens_ds),
            }
        )
    else:
        metadata["kind"] = "unknown"

    return metadata


def _exported_split_filename(split_name, export_format):
    if export_format == HF_DISK_FORMAT:
        return split_name

    if export_format:
        return f"{split_name}.{export_format}"

    return split_name


def _features_to_jsonable(split_ds):
    features = getattr(split_ds, "features", None)
    if features is None:
        return {}

    return {name: str(feature) for name, feature in features.items()}


def _plugin_identity(plugin):
    cls = plugin.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))

    return str(value)
