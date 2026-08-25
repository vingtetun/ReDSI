import os
import inspect

import datasets

from ...decorators.logging import log_function_start

# Stage outputs
CACHE_DIR = "cache"
STAGE_DIR_FULL = "full"
STAGE_DIR = "stage_"


@log_function_start("plugin")
def load_and_preprocess(run_config, plugin, plugin_params, dataset_limit, disable_caching, num_proc):
    if disable_caching:
        datasets.disable_caching()

    stages = plugin.run()

    dataset_name = plugin.name
    cache_dir = _get_dataset_cache_dir(run_config)
    last_stage_dir = make_stage_out_dir_name(len(stages) - 1, dataset_limit, cache_dir)

    ds = load_preprocessed_from_disk(last_stage_dir) if not disable_caching else None
    if ds is None:
        if hasattr(plugin, 'load') and callable(plugin.load):
            ds = plugin.load(plugin_params, dataset_limit, cache_dir)
        else:
            ds = load_from_remote(dataset_name, run_config.datasets, dataset_limit, cache_dir)
        for index, _ in enumerate(stages):
            ds = run_stage(stages[index], ds, plugin_params, num_proc, cache_dir)

            stage_dir = make_stage_out_dir_name(index, dataset_limit, cache_dir)
            save_preprocessed_to_disk(ds, stage_dir)

    return ds


def run_stage(stage, ds, plugin_params, num_proc, cache_dir):
    if "cache_dir" in inspect.signature(stage).parameters:
        return stage(ds, plugin_params, num_proc, cache_dir=cache_dir)

    return stage(ds, plugin_params, num_proc)


def _get_dataset_cache_dir(run_config):
    explicit_cache_dir = getattr(run_config.arguments, "dataset_cache_dir", None)
    if explicit_cache_dir:
        return os.path.normpath(str(explicit_cache_dir))

    return os.path.normpath(os.path.join(run_config.arguments.out_dir, '..', '..', CACHE_DIR))


def load_from_remote(dataset_name, dataset_config, dataset_limit, cache_dir):
    train_split, validation_split = get_train_validation_split(dataset_config, dataset_limit)
    train = datasets.load_dataset(dataset_name, split=train_split, cache_dir=cache_dir)
    validation = datasets.load_dataset(dataset_name, split=validation_split, cache_dir=cache_dir)
    return datasets.DatasetDict(train=train, validation=validation)


def load_preprocessed_from_disk(cache_dir):
    try:
        return datasets.load_from_disk(cache_dir) if os.path.exists(cache_dir) else None
    except:
        return None



def save_preprocessed_to_disk(ds, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    ds.save_to_disk(cache_dir)


def make_stage_out_dir_name(stage_index, dataset_limit, cache_dir):
    name = STAGE_DIR_FULL if dataset_limit <= 0 else str(dataset_limit)
    return os.path.join(cache_dir, f"{STAGE_DIR}{stage_index}", name)


def get_train_validation_split(dataset_config, dataset_limit):
    train_split = dataset_config.train
    validation_split = dataset_config.validation
    if dataset_limit > 0:
        train_split = f"{train_split}[:{dataset_limit}]"
        validation_split = f"{validation_split}[:{dataset_limit}]"
    return train_split, validation_split
