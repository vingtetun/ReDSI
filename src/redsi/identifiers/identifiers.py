import datasets

from ..decorators.logging import log_function_start
from ..formats import formats
from ..formats.results import IdentifierResult, apply_stage_result

STR_ASSIGN_IDENTIFIERS = "Assign identifiers ({split_name})"
KWARGS_IDENTIFIERS = "identifiers"
ERR_UNSUPPORTED_PLUGIN_RESULT = "Unsupported identifier plugin result type: {type_result}"


@log_function_start("plugin")
def assign_identifiers(run_config, plugin, plugin_params, data):
    seed = run_config.arguments.seed

    if isinstance(data, list):
        out = []
        for ds_split_name, ds_split in data:
            result = _normalize_identifier_result(plugin.run(ds_split[formats.KEY_DOCUMENTS], plugin_params, seed))
            updated = apply_identifier_result(ds_split, result, STR_ASSIGN_IDENTIFIERS)
            out.append((ds_split_name, updated))
        return out

    result = _normalize_identifier_result(plugin.run(data[formats.KEY_DOCUMENTS], plugin_params, seed))
    return apply_identifier_result(data, result, STR_ASSIGN_IDENTIFIERS)


def _normalize_identifier_result(result):
    if isinstance(result, IdentifierResult):
        return result

    if isinstance(result, dict):
        return IdentifierResult(docid_map=result)

    msg = ERR_UNSUPPORTED_PLUGIN_RESULT.format(type_result=type(result))
    raise TypeError(msg)


def apply_identifier_result(ds, result, desc_template):
    ds = assign_docids_to_dataset_dict(ds, result.docid_map, desc_template)
    ds = apply_stage_result(ds, result)
    return ds


def assign_docids_to_dataset_dict(ds, docid_map, desc_template):
    out = {}

    for split_name in ds:
        split_ds = ds[split_name]

        if formats.FEATURE_DOCID not in split_ds.column_names:
            out[split_name] = split_ds
            continue

        out[split_name] = split_ds.map(
            _assign_identifiers_batch,
            desc=desc_template.format(split_name=split_name),
            batched=True,
            fn_kwargs={KWARGS_IDENTIFIERS: docid_map},
        )

    return datasets.DatasetDict(**out)


def _assign_identifiers_batch(batch, **kwargs):
    docids = kwargs[KWARGS_IDENTIFIERS]
    keys = batch[formats.FEATURE_DOCID]
    return {formats.FEATURE_DOCID: [docids[key] for key in keys]}
