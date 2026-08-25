import hashlib
import json
import os

import datasets
from omegaconf import DictConfig, OmegaConf

from ..augmentation import augment
from ..dataset.loader import load_and_preprocess
from ..dataset.splits_builder import splits
from ..dataset.stats import stats
from ..export import export
from ..identifiers import assign_identifiers
from ..plugins import normalize_params, plugin_types
from ..transformation import transform
from .setup_tqdm import setup_tqdm

# Errors
ERR_UNSUPPORTED_TYPE = "[{step}] The output type '{out_type}' is unsupported."
ERR_PLUGIN_SECTION_MUST_BE_MAPPING = "[{step}] Expected a mapping of plugin_name -> params, got '{value_type}'."
ERR_PLUGIN_PARAMS_MUST_BE_MAPPING = "[{step}:{plugin}] Expected plugin params to be a mapping, got '{value_type}'."
ERR_DATASET_SECTION_MUST_BE_MAPPING = (
    "[{step}] Expected a mapping containing 'plugin' and optional 'params', got '{value_type}'."
)
ERR_DATASET_PLUGIN_MISSING = "[{step}] Missing required field 'plugin'."
ERR_DATASET_PARAMS_MUST_BE_MAPPING = "[{step}] Expected 'params' to be a mapping, got '{value_type}'."


def _resolve_num_proc(num_proc):
    if num_proc is None:
        cpu = os.cpu_count() or 4
        num_proc = max(1, cpu - 1)  # leave one core free
    elif num_proc <= 1:
        num_proc = None

    return num_proc


def resolve_arguments(args_config):
    args_config.num_proc = _resolve_num_proc(args_config.num_proc)


def is_none_plugin(plugin_name):
    return plugin_name == "none"


def is_dataset_step(step):
    return step == plugin_types.PLUGIN_TYPE_DATASETS


def is_plugin_map_step(step):
    return step in {
        plugin_types.PLUGIN_TYPE_IDENTIFIERS,
        plugin_types.PLUGIN_TYPE_TRANSFORMATION,
        plugin_types.PLUGIN_TYPE_AUGMENTATION,
        plugin_types.PLUGIN_TYPE_EXPORT,
    }


def is_plugin_map(step, step_value):
    return isinstance(step_value, DictConfig) and is_plugin_map_step(step)


def resolve_dataset_plugin(step, step_value):
    if not isinstance(step_value, DictConfig):
        msg = ERR_DATASET_SECTION_MUST_BE_MAPPING.format(
            step=step,
            value_type=type(step_value),
        )
        raise TypeError(msg)

    if "plugin" not in step_value:
        raise ValueError(ERR_DATASET_PLUGIN_MISSING.format(step=step))

    plugin_name = step_value.plugin
    if is_none_plugin(plugin_name):
        return None, {}

    plugin_params = step_value.get("params", {})
    if plugin_params is None:
        plugin_params = {}

    if not isinstance(plugin_params, (DictConfig, dict)):
        msg = ERR_DATASET_PARAMS_MUST_BE_MAPPING.format(
            step=step,
            value_type=type(plugin_params),
        )
        raise TypeError(msg)

    return plugin_name, plugin_params


def iter_plugin_steps(step, step_value):
    if not isinstance(step_value, DictConfig):
        msg = ERR_PLUGIN_SECTION_MUST_BE_MAPPING.format(
            step=step,
            value_type=type(step_value),
        )
        raise TypeError(msg)

    for plugin_name, plugin_params in step_value.items():
        if is_none_plugin(plugin_name):
            continue

        if plugin_params is None:
            plugin_params = {}

        if not isinstance(plugin_params, (DictConfig, dict)):
            msg = ERR_PLUGIN_PARAMS_MUST_BE_MAPPING.format(
                step=step,
                plugin=plugin_name,
                value_type=type(plugin_params),
            )
            raise TypeError(msg)

        yield plugin_name, plugin_params


def run_config(cfg, plugins):
    setup_tqdm()
    resolve_arguments(cfg.arguments)

    out = None
    available_plugins_by_type = plugins.asdict()

    for step, step_value in cfg.items():
        if is_dataset_step(step):
            plugin_name, raw_params = resolve_dataset_plugin(step, step_value)
            if plugin_name is None:
                continue

            available_plugins = available_plugins_by_type.get(step, {})
            plugin = available_plugins.get(plugin_name)
            plugin_params = normalize_params(
                f"{step}:{plugin_name}",
                plugin,
                raw_params,
            )

            out = load_and_preprocess(
                cfg,
                plugin,
                plugin_params,
                cfg.arguments.dataset_limit,
                cfg.arguments.disable_caching,
                cfg.arguments.num_proc,
            )
            out = splits(out, cfg)
            stats(out, cfg)
            _check_output_type(step, out)
            continue

        if not is_plugin_map(step, step_value):
            continue

        available_plugins = available_plugins_by_type.get(step, {})

        for plugin_name, raw_params in iter_plugin_steps(step, step_value):
            plugin = available_plugins.get(plugin_name)
            plugin_params = normalize_params(
                f"{step}:{plugin_name}",
                plugin,
                raw_params,
            )

            if step == plugin_types.PLUGIN_TYPE_IDENTIFIERS:
                out = assign_identifiers(cfg, plugin, plugin_params, out)

            elif step == plugin_types.PLUGIN_TYPE_TRANSFORMATION:
                out = transform(cfg, plugin, plugin_params, out)

            elif step == plugin_types.PLUGIN_TYPE_AUGMENTATION:
                out = augment(cfg, plugin, plugin_params, out)

            elif step == plugin_types.PLUGIN_TYPE_EXPORT:
                export(cfg, plugin, plugin_params, out)

            else:
                continue

            if step != plugin_types.PLUGIN_TYPE_EXPORT:
                _check_output_type(step, out)


def _check_output_type(step, out):
    if isinstance(out, datasets.DatasetDict):
        return

    if isinstance(out, list):
        for _, ds_split in out:
            if not isinstance(ds_split, datasets.DatasetDict):
                msg = ERR_UNSUPPORTED_TYPE.format(step=step, out_type=str(type(ds_split)))
                raise TypeError(msg)
        return

    msg = ERR_UNSUPPORTED_TYPE.format(step=step, out_type=str(type(out)))
    raise TypeError(msg)


def sha1_of_value(value):
    container = OmegaConf.to_container(value, resolve=True)
    encoded = json.dumps(
        container,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha1(encoded).hexdigest()


def plugin_names(cfg_section):
    if cfg_section is None:
        return ""

    if isinstance(cfg_section, DictConfig) and "plugin" in cfg_section:
        plugin_name = cfg_section.plugin
        return "" if plugin_name is None else str(plugin_name)

    if isinstance(cfg_section, (DictConfig, dict)):
        return "_".join(cfg_section.keys())

    return ""


OmegaConf.register_new_resolver("sha1_value", sha1_of_value)
OmegaConf.register_new_resolver("plugin_names", plugin_names)
