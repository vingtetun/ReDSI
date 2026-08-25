from pathlib import Path

from redsi.builtins.datasets.nq320k.package import write_package
from redsi.plugins import Param

PLUGIN_DESCRIPTION = "Write packaged NQ320K documents and query splits as gzip JSONL."
PLUGIN_PARAMS = [
    Param(
        "output_subdir",
        "str",
        default="nq320k_package",
        help="Directory to write the package into, relative to the export output directory unless absolute.",
    ),
    Param(
        "text_style",
        "str",
        default="",
        help="Optional text style label recorded in package metadata.",
    ),
    Param(
        "dedup_method",
        "str",
        default="",
        help="Optional dedup/grouping label recorded in package metadata.",
    ),
]


class NQ320KPackageExportPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params, out_dir):
        package_dir = resolve_output_dir(out_dir, params.output_subdir)
        metadata = {
            "text_style": params.text_style or None,
            "dedup_method": params.dedup_method or None,
            "docid_scope": "variant-local-sequential",
        }
        write_package(ds, package_dir, metadata=metadata)


def resolve_output_dir(out_dir, output_subdir):
    output_path = Path(output_subdir).expanduser()
    if output_path.is_absolute():
        return output_path
    return Path(out_dir) / output_path
