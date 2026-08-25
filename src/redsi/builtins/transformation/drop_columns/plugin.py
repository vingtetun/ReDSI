import datasets

from ....plugins import Param

PLUGIN_DESCRIPTION = "Drop configured columns from selected dataset splits."

PLUGIN_PARAMS = [
    Param(
        "columns",
        "list",
        default=[],
        help="Column names to remove.",
    ),
    Param(
        "splits",
        "list",
        default=[],
        help="Split names to update. If empty, update all splits.",
    ),
    Param(
        "ignore_missing",
        "bool",
        default=True,
        help="If True, ignore missing splits and columns. If False, raise an error.",
    ),
]

ERR_INVALID_DATASET = "drop_columns: expected a DatasetDict, got {dataset_type}"
ERR_MISSING_SPLIT = "drop_columns: missing split '{split_name}'"
ERR_MISSING_COLUMNS = "drop_columns: split '{split_name}' missing columns {columns}"


class DropColumnsPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params):
        if not isinstance(ds, datasets.DatasetDict):
            raise TypeError(ERR_INVALID_DATASET.format(dataset_type=type(ds)))

        columns = _normalize_names(params.columns)
        if not columns:
            return ds

        split_names = _normalize_names(params.splits)
        target_split_names = split_names or list(ds.keys())
        ignore_missing = bool(params.ignore_missing)

        out = dict(ds)
        for split_name in target_split_names:
            if split_name not in ds:
                if ignore_missing:
                    continue
                raise ValueError(ERR_MISSING_SPLIT.format(split_name=split_name))

            split_ds = ds[split_name]
            available_columns = set(split_ds.column_names)
            columns_to_drop = [column for column in columns if column in available_columns]
            missing_columns = [column for column in columns if column not in available_columns]

            if missing_columns and not ignore_missing:
                raise ValueError(
                    ERR_MISSING_COLUMNS.format(
                        split_name=split_name,
                        columns=missing_columns,
                    )
                )

            if columns_to_drop:
                out[split_name] = split_ds.remove_columns(columns_to_drop)

        return datasets.DatasetDict(out)


def _normalize_names(values):
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    return [str(value) for value in values if str(value)]


plugin = DropColumnsPlugin()
