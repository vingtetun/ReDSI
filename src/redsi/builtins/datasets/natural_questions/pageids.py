import html
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

KEY_PARAMETER_TITLE = "title"
KEY_PARAMETER_OLDID = "oldid"
KEY_PARAMETER_PAGEID = "pageid"

OLDID_TO_PAGEID_FILEPATH = "data/oldid_to_pageid.jsonl"

_MAPPINGS = {}


def setup():
    path = Path(os.path.dirname(__file__), OLDID_TO_PAGEID_FILEPATH)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            _MAPPINGS[str(row[KEY_PARAMETER_OLDID])] = row[KEY_PARAMETER_PAGEID]


def url_to_page_id(url):
    url = html.unescape(url)
    parts = urlsplit(url)
    qs = parse_qs(parts.query)

    oldid = qs[KEY_PARAMETER_OLDID][0]
    if oldid in _MAPPINGS:
        pageid = _MAPPINGS[oldid]
    else:
        title = qs[KEY_PARAMETER_TITLE][0]
        pageid = hash(title)

    return pageid
