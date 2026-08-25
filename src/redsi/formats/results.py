from dataclasses import dataclass, field

import datasets

ERR_SPLIT_EXISTS = "Cannot add extra split '{split_name}': split already exists"
ERR_UPDATED_SPLIT_MISSING = "Cannot update split '{split_name}': split does not exist"
ERR_SPLIT_WRONG_TYPE = "Split '{split_name}' must be a datasets.Dataset, got {split_type}"


@dataclass
class StageResult:
    updated_splits: dict[str, datasets.Dataset] = field(default_factory=dict)
    extra_splits: dict[str, datasets.Dataset] = field(default_factory=dict)


@dataclass
class IdentifierResult(StageResult):
    docid_map: dict[str, str] = field(default_factory=dict)


def apply_stage_result(ds, result):
    return _merge_stage_splits(ds, result.updated_splits, result.extra_splits)


def _merge_stage_splits(ds, updated_splits, extra_splits):
    if not updated_splits and not extra_splits:
        return ds

    out = dict(ds)

    for split_name, split_ds in updated_splits.items():
        if split_name not in out:
            msg = ERR_UPDATED_SPLIT_MISSING.format(split_name=split_name)
            raise ValueError(msg)

        _validate_split(split_name, split_ds)
        out[split_name] = split_ds

    for split_name, split_ds in extra_splits.items():
        if split_name in out:
            msg = ERR_SPLIT_EXISTS.format(split_name=split_name)
            raise ValueError(msg)

        _validate_split(split_name, split_ds)
        out[split_name] = split_ds

    return datasets.DatasetDict(out)


def _validate_split(split_name, split_ds):
    if not isinstance(split_ds, datasets.Dataset):
        msg = ERR_SPLIT_WRONG_TYPE.format(split_name=split_name, split_type=type(split_ds))
        raise TypeError(msg)
