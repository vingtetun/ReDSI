from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import datasets

from redsi.formats.results import StageResult
from redsi.plugins import Param

###############################################################################
# Constants
###############################################################################

FEATURE_DOCID = "docid"
FEATURE_TEXT = "text"
FEATURE_TITLE = "title"

SPLIT_DOCUMENTS = "documents"
SPLIT_TITLES = "titles"

ERR_MISSING_DOCUMENTS_SPLIT = "titles augmentation: missing '{split_name}' split"
ERR_NO_DOCID_COLUMN = "titles augmentation: missing required column '{column_name}'"
ERR_NO_TITLE_COLUMN = "titles augmentation: missing required column '{column_name}'"

PLUGIN_DESCRIPTION = (
    "Emit a compact extra split named 'titles' from document titles only. "
    "Each emitted row contains {docid, text}, where text is the document title."
)

PLUGIN_PARAMS = [
    Param(
        "lowercase",
        "bool",
        default=True,
        help="Lowercase emitted titles.",
    ),
]

###############################################################################
# Helpers
###############################################################################


def clean_text(text, lowercase=False):
    if text is None:
        return ""
    text = " ".join(str(text).split()).strip()
    if lowercase:
        text = text.lower()
    return text


def build_title_rows(ds, lowercase=False):
    column_names = set(ds.column_names)

    if FEATURE_DOCID not in column_names:
        raise ValueError(ERR_NO_DOCID_COLUMN.format(column_name=FEATURE_DOCID))

    if FEATURE_TITLE not in column_names:
        raise ValueError(ERR_NO_TITLE_COLUMN.format(column_name=FEATURE_TITLE))

    rows = []
    for row in ds:
        docid = row.get(FEATURE_DOCID)
        title = clean_text(row.get(FEATURE_TITLE), lowercase=lowercase)

        if not title:
            continue

        rows.append(
            {
                FEATURE_DOCID: docid,
                FEATURE_TEXT: title,
            }
        )

    return rows


###############################################################################
# Plugin
###############################################################################


class TitlesAugmentationPlugin:
    description = PLUGIN_DESCRIPTION
    params = PLUGIN_PARAMS

    def run(self, ds, params, seed):
        del seed

        if SPLIT_DOCUMENTS not in ds:
            raise ValueError(ERR_MISSING_DOCUMENTS_SPLIT.format(split_name=SPLIT_DOCUMENTS))

        rows = build_title_rows(
            ds[SPLIT_DOCUMENTS],
            lowercase=bool(params.lowercase),
        )

        titles_ds = datasets.Dataset.from_list(rows)

        return StageResult(
            extra_splits={
                SPLIT_TITLES: titles_ds,
            }
        )


plugin = TitlesAugmentationPlugin()
