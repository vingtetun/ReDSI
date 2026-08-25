from ..decorators.logging import log_function_start


@log_function_start("datasets.plugin", "plugin")
def transform(run_config, plugin, plugin_params, data):
    if isinstance(data, list):
        out = []
        for ds_split_name, ds_split in data:
            out.append((ds_split_name, plugin.run(ds_split, plugin_params)))
    else:
        out = plugin.run(data, plugin_params)

    return out
