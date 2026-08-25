import datasets

from ....formats import formats
from ....plugins import Param

PLUGIN_DESCRIPTION = (
    "Transform the dataset into a multitask layout by merging documents and "
    "train queries into a single train split, and renaming validation queries "
    "to validation. This transform consumes the source splits and emits "
    "Hugging Face seq2seq token column names."
)

PLUGIN_PARAMS = [
    Param(
        "train_output_split",
        "str",
        default="train",
        help="Name of the output train split.",
    ),
    Param(
        "validation_output_split",
        "str",
        default="validation",
        help="Name of the output validation split.",
    ),
    Param(
        "shuffle_train",
        "bool",
        default=False,
        help="Shuffle the merged train split.",
    ),
    Param(
        "shuffle_validation",
        "bool",
        default=False,
        help="Shuffle the validation split.",
    ),
    Param(
        "shuffle_seed",
        "int",
        default=42,
        help="Seed used when shuffling splits.",
    ),
    Param(
        "document_row_multiplier",
        "int",
        default=1,
        help="Number of times document rows are repeated in the output train split.",
    ),
]

ERR_OUTPUT_SPLIT_CONFLICT = (
    "multitask transform: output split '{split_name}' already exists and is not one of the source splits being consumed"
)
ERR_DOCUMENT_ROW_MULTIPLIER = "multitask transform: document_row_multiplier must be >= 1, got {value}"
ERR_RENAME_CONFLICT = "multitask transform: cannot rename '{source}' to '{target}' because '{target}' already exists"

FINAL_TOKEN_COLUMN_RENAMES = {
    formats.FEATURE_TEXT_TOKENS: "input_ids",
    formats.FEATURE_DOCID_TOKENS: "labels",
}


class MultitaskPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params):
        source_split_names = {
            formats.KEY_DOCUMENTS,
            formats.KEY_TRAIN_QUERIES,
            formats.KEY_VALIDATION_QUERIES,
        }

        train_output_split = params.train_output_split
        validation_output_split = params.validation_output_split

        _validate_output_split_name(ds, train_output_split, source_split_names)
        _validate_output_split_name(ds, validation_output_split, source_split_names)

        train_documents = _repeat_documents(
            ds[formats.KEY_DOCUMENTS],
            params.document_row_multiplier,
        )

        train_ds = datasets.concatenate_datasets(
            [
                train_documents,
                ds[formats.KEY_TRAIN_QUERIES],
            ]
        )
        if params.shuffle_train:
            train_ds = train_ds.shuffle(seed=params.shuffle_seed)

        validation_ds = ds[formats.KEY_VALIDATION_QUERIES]
        if params.shuffle_validation:
            validation_ds = validation_ds.shuffle(seed=params.shuffle_seed)

        ds[train_output_split] = train_ds
        ds[validation_output_split] = validation_ds

        for split_name in source_split_names:
            if split_name not in {train_output_split, validation_output_split}:
                del ds[split_name]

        for split_name in list(ds.keys()):
            ds[split_name] = _rename_final_token_columns(ds[split_name])

        return ds


def _validate_output_split_name(ds, split_name, source_split_names):
    if split_name in ds and split_name not in source_split_names:
        raise ValueError(ERR_OUTPUT_SPLIT_CONFLICT.format(split_name=split_name))


def _repeat_documents(documents, document_row_multiplier):
    document_row_multiplier = int(document_row_multiplier)
    if document_row_multiplier < 1:
        raise ValueError(ERR_DOCUMENT_ROW_MULTIPLIER.format(value=document_row_multiplier))

    if document_row_multiplier == 1:
        return documents

    return datasets.concatenate_datasets([documents] * document_row_multiplier)


def _rename_final_token_columns(ds):
    column_names = set(ds.column_names)
    rename_columns = {}

    for source, target in FINAL_TOKEN_COLUMN_RENAMES.items():
        if source not in column_names:
            continue

        if target in column_names:
            raise ValueError(ERR_RENAME_CONFLICT.format(source=source, target=target))

        rename_columns[source] = target

    if not rename_columns:
        return ds

    return ds.rename_columns(rename_columns)


plugin = MultitaskPlugin()
