import sys

import hydra

from redsi import config_dir, discover_plugins, run_config


def early_flags(argv):
    if "--plugins" in argv:
        plugins = discover_plugins()
        print(plugins)
        sys.exit(0)


@hydra.main(version_base=None, config_path=config_dir(), config_name="dsi")
def main(config):
    plugins = discover_plugins()
    run_config(config, plugins)


def cli():
    early_flags(sys.argv)
    main()
