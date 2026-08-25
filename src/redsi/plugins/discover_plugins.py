from .discover_group import discover_group
from .plugin_types import PLUGIN_TYPES
from .plugins import Plugins


def discover_plugins():
    package_name = __package__
    groups = {key: discover_group(f"{package_name}.{key}") for key in PLUGIN_TYPES}
    return Plugins(groups)
