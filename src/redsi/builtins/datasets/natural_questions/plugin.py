from redsi.plugins import Param

PLUGIN_DESCRIPTION = "Google Natural Questions dataset with built-in preprocessing stages."
PLUGIN_PARAMS = [
    Param(
        "data_source",
        "str",
        default=None,
        help="A directory or a url where the simplified versions of the dataset lives.",
    ),
    Param(
        "text_style",
        "str",
        default="simplenorm",
        help="",
    ),
    Param(
        "text_limit",
        "int",
        default=4000,
        help="",
    ),
    Param(
        "dedup_method",
        "str",
        default="url",
        help="",
    ),
    Param(
        "lowercase",
        "bool",
        default=True,
        help="",
    ),
]


class NaturalQuestionsDatasetPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def load(self, params, dataset_limit, cache_dir):
        from .loader import load

        return load(params, dataset_limit, cache_dir)

    def run(self):
        from .dataset import stages

        return stages
