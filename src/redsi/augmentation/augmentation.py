from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import datasets

from ..decorators.logging import log_function_start
from ..formats.results import StageResult, apply_stage_result

ERR_UNSUPPORTED_AUGMENT_RESULT = "augmentation: unsupported plugin result type '{result_type}'"
ERR_EXTRA_SPLIT_ALREADY_EXISTS = "augmentation: cannot add extra split '{split_name}': split already exists"
ERR_EXTRA_SPLIT_INVALID_TYPE = "augmentation: extra split '{split_name}' must be a datasets.Dataset, got {split_type}"

CACHE_DIR = "cache"
CACHE_SUBDIR = "augment"
CACHE_VERSION = "v1"


@log_function_start("datasets.plugin", "plugin")
def augment(run_config, plugin, plugin_params, data):
    seed = run_config.arguments.seed

    cache_root = _get_augmentation_cache_root(run_config)
    use_cache = bool(cache_root)

    if isinstance(data, list):
        out = []
        for ds_split_name, ds_split in data:
            if use_cache:
                cached = _try_load_cached_augmented_dataset(
                    cache_root=cache_root,
                    plugin=plugin,
                    plugin_params=plugin_params,
                    seed=seed,
                    data=ds_split,
                    split_name=ds_split_name,
                )
                if cached is not None:
                    out.append((ds_split_name, cached))
                    continue

            result = plugin.run(ds_split, plugin_params, seed)
            result = _normalize_stage_result(result)
            updated_ds = apply_stage_result(ds_split, result)

            if use_cache:
                _save_cached_augmented_dataset(
                    cache_root=cache_root,
                    plugin=plugin,
                    plugin_params=plugin_params,
                    seed=seed,
                    data=ds_split,
                    split_name=ds_split_name,
                    augmented_dataset=updated_ds,
                )

            out.append((ds_split_name, updated_ds))
        return out

    if use_cache:
        cached = _try_load_cached_augmented_dataset(
            cache_root=cache_root,
            plugin=plugin,
            plugin_params=plugin_params,
            seed=seed,
            data=data,
            split_name=None,
        )
        if cached is not None:
            return cached

    result = plugin.run(data, plugin_params, seed)
    result = _normalize_stage_result(result)
    updated_ds = apply_stage_result(data, result)

    if use_cache:
        _save_cached_augmented_dataset(
            cache_root=cache_root,
            plugin=plugin,
            plugin_params=plugin_params,
            seed=seed,
            data=data,
            split_name=None,
            augmented_dataset=updated_ds,
        )

    return updated_ds


def _normalize_stage_result(result):
    if isinstance(result, StageResult):
        return result

    msg = ERR_UNSUPPORTED_AUGMENT_RESULT.format(result_type=str(type(result)))
    raise TypeError(msg)


def _get_augmentation_cache_root(run_config):
    if run_config.arguments.disable_caching:
        return None

    explicit_cache_dir = getattr(run_config.arguments, "augmentation_cache_dir", None)
    if explicit_cache_dir:
        return os.path.normpath(str(explicit_cache_dir))

    cache_dir = os.path.normpath(os.path.join(run_config.arguments.out_dir, "..", "..", CACHE_DIR))
    return os.path.join(cache_dir, CACHE_SUBDIR)


def _try_load_cached_augmented_dataset(
    cache_root,
    plugin,
    plugin_params,
    seed,
    data,
    split_name,
):
    cache_path = _build_cache_path(
        cache_root=cache_root,
        plugin=plugin,
        plugin_params=plugin_params,
        seed=seed,
        data=data,
        split_name=split_name,
    )

    if not os.path.isdir(cache_path):
        return None

    cached = datasets.load_from_disk(cache_path)
    return _restore_non_cached_splits_from_input(cached=cached, current=data)


IDENTIFIER_OWNED_SPLITS = {
    "tokens",
    "tokens_stage_2",
    "cluster_centroids",
    "cluster_assignments",
    "cluster_index",
}

IDENTIFIER_OWNED_PREFIXES = (
    "tokens_",
    "cluster_",
    "online_",
)


def _is_identifier_owned_split(split_name):
    split_name = str(split_name)

    if split_name in IDENTIFIER_OWNED_SPLITS:
        return True

    return any(split_name.startswith(prefix) for prefix in IDENTIFIER_OWNED_PREFIXES)


def _restore_non_cached_splits_from_input(cached, current):
    if not isinstance(cached, datasets.DatasetDict):
        return cached

    if not isinstance(current, datasets.DatasetDict):
        return cached

    merged = {name: ds for name, ds in cached.items()}

    # Keep fresh identifier-owned splits from the current dataset.
    # This matters because augmentation cache may be valid while identifier outputs
    # such as tokens / cluster_centroids / cluster_index changed upstream.
    for split_name, split in current.items():
        if _is_identifier_owned_split(split_name):
            merged[split_name] = split

    return datasets.DatasetDict(merged)


def _save_cached_augmented_dataset(
    cache_root,
    plugin,
    plugin_params,
    seed,
    data,
    split_name,
    augmented_dataset,
):
    cache_path = _build_cache_path(
        cache_root=cache_root,
        plugin=plugin,
        plugin_params=plugin_params,
        seed=seed,
        data=data,
        split_name=split_name,
    )

    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)

    tmp_path = f"{cache_path}.tmp"
    if os.path.exists(tmp_path):
        _remove_path(tmp_path)

    augmented_dataset.save_to_disk(tmp_path)

    if os.path.exists(cache_path):
        _remove_path(cache_path)

    os.replace(tmp_path, cache_path)


def _build_cache_path(
    cache_root,
    plugin,
    plugin_params,
    seed,
    data,
    split_name,
):
    plugin_name = _plugin_identity(plugin)
    cache_key = _compute_cache_key(
        plugin=plugin,
        plugin_params=plugin_params,
        seed=seed,
        data=data,
        split_name=split_name,
    )
    safe_plugin_name = plugin_name.replace("/", "_").replace(".", "_")
    return os.path.join(cache_root, safe_plugin_name, cache_key)


def _compute_cache_key(plugin, plugin_params, seed, data, split_name):
    payload = {
        "cache_version": CACHE_VERSION,
        "plugin": _plugin_identity(plugin),
        "plugin_params": _to_stable_jsonable(plugin_params),
        "seed": int(seed),
        "split_name": split_name,
        "input_dataset": _dataset_identity(data),
    }

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _plugin_identity(plugin):
    cls = plugin.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _dataset_identity(data):
    """
    Prefer Hugging Face dataset fingerprints.
    Fall back to lightweight structural info if needed.
    """
    fingerprint = getattr(data, "_fingerprint", None)
    if fingerprint is not None:
        return {
            "type": type(data).__name__,
            "fingerprint": str(fingerprint),
        }

    # Fallback for unusual dataset-like objects
    identity = {
        "type": type(data).__name__,
    }

    column_names = getattr(data, "column_names", None)
    if column_names is not None:
        identity["column_names"] = list(column_names)

    try:
        identity["num_rows"] = len(data)
    except Exception:
        pass

    return identity


def _to_stable_jsonable(value):
    """
    Convert params into a deterministic JSON-serializable structure.
    Works for simple objects, dataclasses, argparse/Hydra-ish namespaces, dicts, etc.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_stable_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}

    if isinstance(value, (list, tuple)):
        return [_to_stable_jsonable(v) for v in value]

    if isinstance(value, set):
        return sorted(_to_stable_jsonable(v) for v in value)

    if hasattr(value, "__dict__"):
        return _to_stable_jsonable(vars(value))

    return repr(value)


def _remove_path(path):
    if os.path.isdir(path):
        import shutil

        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
