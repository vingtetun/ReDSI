import re

from . import features, titles
from .htmlnorm import extract_text

# Text extraction methods
TEXT_STYLE_SIMPLIFIED = "simplified"
TEXT_STYLE_SIMPLENORM = "simplenorm"
TEXT_STYLE_TITLE_ABSTRACT_CONTENT = "title-abstract-content"
TEXT_STYLE_NCINORM = "ncinorm"
TEXT_STYLE_HTMLNORM = "htmlnorm"
TEXT_STYLE_HTMLNORM_INFOBOX = "htmlnorm-infobox"

# HTML Tags
H1_OPEN = "<H1>"
H1_CLOSE = "</H1>"
P_OPEN = "<P>"
P_CLOSE = "</P>"
UL_CLOSE = "</Ul>"

# Splits
TRAIN_SPLIT_NAME = "train"

# Errors
ERR_UNSUPPORTED_TEXT_STYLE = "The text style method {name} is unsupported."

_RE_HTML_TAG = re.compile(r"<[^>]+>")

_HTMLNORM_TOKENIZER = None


def get_htmlnorm_tokenizer():
    global _HTMLNORM_TOKENIZER

    if _HTMLNORM_TOKENIZER is None:
        from transformers import AutoTokenizer

        _HTMLNORM_TOKENIZER = AutoTokenizer.from_pretrained("google-t5/t5-small")

    return _HTMLNORM_TOKENIZER


def strip_html(text):
    return " ".join(_RE_HTML_TAG.sub(" ", text or "").split())


def get_content(row):
    text = row[features.KEY_DOCUMENT_TEXT]
    return text


def get_content_simplenorm(row):
    text = get_content(row)
    return strip_html(text)


def get_content_title_abstract_content(row):
    return get_content_title_abstract_content_base(row)


def get_content_ncinorm(row, split_name=None):
    text = get_content(row)
    title, _ = _extract_between(text, H1_OPEN, H1_CLOSE)

    if split_name is not None and split_name != TRAIN_SPLIT_NAME:
        title = get_document_title(row) or title

    return get_content_title_abstract_content_base(row, title=title)


def get_content_title_abstract_content_base(row, title=None):
    text = get_content(row)
    if title is None:
        title, _ = _extract_between(text, H1_OPEN, H1_CLOSE)
    title = title.strip()
    abstract, abstract_end_index = _extract_between(text, P_OPEN, P_CLOSE)
    content = _extract_content_after_abstract(text, abstract_end_index)
    content = strip_html(content)

    return title + abstract + content


def get_content_htmlnorm_base(row, remove_infobox):
    tokenizer = get_htmlnorm_tokenizer()
    content = extract_text(row, tokenizer, remove_infobox)

    return content


def get_content_htmlnorm(row):
    return get_content_htmlnorm_base(row, True)


def get_content_htmlnorm_infobox(row):
    return get_content_htmlnorm_base(row, False)


TEXT_STYLES = {
    TEXT_STYLE_SIMPLIFIED: get_content,
    TEXT_STYLE_SIMPLENORM: get_content_simplenorm,
    TEXT_STYLE_TITLE_ABSTRACT_CONTENT: get_content_title_abstract_content,
    TEXT_STYLE_NCINORM: get_content_ncinorm,
    TEXT_STYLE_HTMLNORM: get_content_htmlnorm,
    TEXT_STYLE_HTMLNORM_INFOBOX: get_content_htmlnorm_infobox,
}


def get_text_style_method_from_name(text_style):
    method = TEXT_STYLES.get(text_style)
    if method is None:
        raise ValueError(ERR_UNSUPPORTED_TEXT_STYLE.format(name=text_style))

    return method


def get_text_style_method(config):
    return get_text_style_method_from_name(config.text_style)


def preload_text_style_resources(text_style):
    if text_style in {
        TEXT_STYLE_HTMLNORM,
        TEXT_STYLE_HTMLNORM_INFOBOX,
    }:
        get_htmlnorm_tokenizer()


def apply_text_style(method, row, split_name=None):
    if method is get_content_ncinorm:
        return method(row, split_name=split_name)
    return method(row)


# Helpers
def get_document_title(row):
    title = row.get(features.KEY_DOCUMENT_TITLE)
    if title:
        return title

    url = row.get(features.KEY_DOCUMENT_URL)
    if not url:
        return ""

    try:
        return titles.from_url(url)
    except (KeyError, IndexError, ValueError):
        return ""


def _extract_between(text, start_tag, end_tag):
    try:
        start = text.index(start_tag) + len(start_tag)
        end = text.index(end_tag, start)
        return text[start:end], end
    except ValueError:
        return "", -1


def _extract_content_after_abstract(text, abstract_end_index):
    if abstract_end_index == -1:
        return ""

    start = abstract_end_index + len(P_CLOSE)

    first_final = text.rfind(UL_CLOSE)
    if first_final == -1:
        end = len(text)
    else:
        text_before_last_ul = text[:first_final]
        second_final = text_before_last_ul.rfind(UL_CLOSE)
        end = second_final if second_final != -1 else first_final

    if start >= end:
        return ""

    return text[start:end]


###############################################################################
# Text style requirements
###############################################################################


def requires_full_format(text_style):
    return text_style in {
        TEXT_STYLE_HTMLNORM,
        TEXT_STYLE_HTMLNORM_INFOBOX,
    }


def requires_simplified_format(text_style):
    return not requires_full_format(text_style)
