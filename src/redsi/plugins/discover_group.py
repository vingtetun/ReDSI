import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from . import plugin_definition

# Errors
ERR_PLUGIN_DUPLICATE = "[{identifier}] is duplicated."


def discover_group(group):
    found = {}

    eps = entry_points()
    selected = eps.select(group=group)
    for ep in selected:
        obj = ep.load()
        plugin = obj()
        plugin.name = ep.name

        identifier = f"{group}.{ep.name}"
        plugin_definition.check_plugin(identifier, plugin)

        if plugin.name in found:
            msg = ERR_PLUGIN_DUPLICATE.format(identifier=identifier)
            raise ValueError(msg)

        found[plugin.name] = plugin

    return found
