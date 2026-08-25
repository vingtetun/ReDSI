from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset dataset with built-in preprocessing stages."
)
PLUGIN_PARAMS = [
    Param(
        "data_source",
        "str",
        default=None,
        help="A directory or a url where the versions of the dataset lives.",
    ),
    Param(
        "lowercase",
        "bool",
        default=True,
        help="",
    ),
    Param(
        "text_style",
        "str",
        default="content",
        help="Document text style: content or title-content.",
    ),
]


class MSMarcoDocumentsDatasetPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def load(self, params, dataset_limit, cache_dir):
        from .loader import load

        return load(params, dataset_limit, cache_dir)

    def run(self):
        from .dataset import stages

        return stages
