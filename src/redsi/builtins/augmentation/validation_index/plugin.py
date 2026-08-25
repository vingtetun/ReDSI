from redsi.formats import formats
from redsi.formats.results import StageResult
from redsi.plugins import Param

PLUGIN_DESCRIPTION = (
    "Create a validation_index split from validation document rows. "
    "After tokenization, rows evaluate document text/input_ids -> docid labels."
)

PLUGIN_PARAMS = [
    Param(
        "output_split_name",
        "str",
        default="validation_index",
        help="Name of the emitted split.",
    ),
]

ERR_MISSING_DOCUMENTS_SPLIT = "validation_index augmentation: missing '{split_name}' split"
ERR_MISSING_VALIDATION_QUERIES_SPLIT = "validation_index augmentation: missing '{split_name}' split"
ERR_MISSING_COLUMNS = "validation_index augmentation: {split_name} split missing columns {columns}"
ERR_MISSING_VALIDATION_DOCUMENTS = "validation_index augmentation: selected {selected}/{expected} validation documents"


class ValidationIndexAugmentationPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params, seed):
        del seed

        if formats.KEY_DOCUMENTS not in ds:
            raise ValueError(ERR_MISSING_DOCUMENTS_SPLIT.format(split_name=formats.KEY_DOCUMENTS))

        if formats.KEY_VALIDATION_QUERIES not in ds:
            raise ValueError(ERR_MISSING_VALIDATION_QUERIES_SPLIT.format(split_name=formats.KEY_VALIDATION_QUERIES))

        documents = ds[formats.KEY_DOCUMENTS]
        validation_queries = ds[formats.KEY_VALIDATION_QUERIES]
        _validate_documents_split(formats.KEY_DOCUMENTS, documents)
        _validate_docid_split(formats.KEY_VALIDATION_QUERIES, validation_queries)

        validation_docids = set(validation_queries[formats.FEATURE_DOCID])
        validation_documents = documents.filter(
            _is_validation_document,
            fn_kwargs={"validation_docids": validation_docids},
            desc="Select validation documents",
        )
        _validate_selected_documents(validation_documents, validation_docids)

        validation_index = validation_documents.map(
            _build_validation_index_batch,
            batched=True,
            remove_columns=validation_documents.column_names,
            desc=f"Build {params.output_split_name}",
        )

        return StageResult(
            extra_splits={
                params.output_split_name: validation_index,
            }
        )


def _validate_documents_split(split_name, documents):
    required_columns = {
        formats.FEATURE_DOCID,
        formats.FEATURE_TEXT,
    }
    _validate_columns(split_name, documents, required_columns)


def _validate_docid_split(split_name, ds):
    _validate_columns(split_name, ds, {formats.FEATURE_DOCID})


def _validate_columns(split_name, ds, required_columns):
    missing_columns = sorted(required_columns - set(ds.column_names))
    if missing_columns:
        raise ValueError(ERR_MISSING_COLUMNS.format(split_name=split_name, columns=missing_columns))


def _is_validation_document(example, validation_docids):
    return example[formats.FEATURE_DOCID] in validation_docids


def _validate_selected_documents(validation_documents, validation_docids):
    expected = len(validation_docids)
    selected = len(validation_documents)
    if selected < expected:
        raise ValueError(
            ERR_MISSING_VALIDATION_DOCUMENTS.format(
                selected=selected,
                expected=expected,
            )
        )


def _build_validation_index_batch(batch):
    size = len(batch[formats.FEATURE_DOCID])
    return {
        formats.FEATURE_DOCID: batch[formats.FEATURE_DOCID],
        formats.FEATURE_KIND: [formats.FEATURE_KIND_DOCUMENT] * size,
        formats.FEATURE_TEXT: batch[formats.FEATURE_TEXT],
    }


plugin = ValidationIndexAugmentationPlugin()
