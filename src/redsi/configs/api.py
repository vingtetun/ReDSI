from importlib.resources import files

from hydra import compose, initialize_config_dir

from redsi.plugins import discover_plugins

from .config_runner import run_config

DEFAULT_CONFIG_NAME = "dsi"


def config_dir():
    return str(files("redsi").joinpath("configs", "yaml"))


def compose_config(config_name=DEFAULT_CONFIG_NAME, overrides=None):
    with initialize_config_dir(config_dir=config_dir(), version_base=None):
        return compose(config_name=config_name, overrides=overrides or [])


def run(config_name=DEFAULT_CONFIG_NAME, overrides=None, plugins=None):
    cfg = compose_config(config_name=config_name, overrides=overrides)
    if plugins is None:
        plugins = discover_plugins()
    return run_config(cfg, plugins)
