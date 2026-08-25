#!/usr/bin/env python3

import argparse
import gzip
import html
import json
import re
import unicodedata

# =============================================================================
# CLI
# =============================================================================

DEFAULT_INPUT = "build/natural_questions/data/nq-dev-all.jsonl.gz"

# =============================================================================
# Regexes
# =============================================================================

TAG_RE = re.compile(r"<[^>]+>")
TAG_NAME_RE = re.compile(r"<\s*/?\s*([a-zA-Z0-9]+)")
ATTRIBUTE_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*["\']([^"\']*)["\']')
WHITESPACE_RE = re.compile(r"\s+")
MANY_NEWLINES_RE = re.compile(r"\n\n+")

# =============================================================================
# Separators / formatting
# =============================================================================

SEPARATOR_EMPTY = ""
SEPARATOR_SPACE = " "
SEPARATOR_LINE_BREAK = "\n"
SEPARATOR_SECTION_BREAK = "\n\n"

INDENT_UNIT = "  "

UNORDERED_LIST_MARKER = "-"
TABLE_CELL_SEPARATOR = " ; "

# =============================================================================
# Wikipedia cleaning
# =============================================================================

DEFAULT_INTRO_PRONUNCIATION_MAX_CHARS = 256
WIKIPEDIA_UI_TEXT_RE = re.compile(r"\[\s*(edit|hide)\s*\]", re.IGNORECASE)

# =============================================================================
# AST keys / kinds
# =============================================================================

KEY_KIND = "kind"
KEY_TAG = "tag"
KEY_TEXT = "text"
KEY_TOKEN = "token"
KEY_ATTRS_HTML = "attrs_html"
KEY_ATTRS_PARSED = "_attrs"
KEY_CHILDREN = "children"
KEY_TITLE = "title"
KEY_LEVEL = "level"
KEY_NODE = "node"
KEY_COLSPAN = "colspan"

KIND_ROOT = "root"
KIND_ELEMENT = "element"
KIND_TEXT = "text"
KIND_SECTION = "section"

# =============================================================================
# Natural Questions row / token keys
# =============================================================================

ROW_EXAMPLE_ID = "example_id"
ROW_DOCUMENT_HTML = "document_html"
ROW_DOCUMENT_TOKENS = "document_tokens"
ROW_DOCUMENT_URL = "document_url"
ROW_QUESTION_TEXT = "question_text"
ROW_DOCUMENT_TEXT = "document_text"
ROW_DOCUMENT_TITLE = "document_title"

TOKEN_HTML_TOKEN = "html_token"
TOKEN_START_BYTE = "start_byte"
TOKEN_END_BYTE = "end_byte"
TOKEN_TEXT = "token"

# =============================================================================
# HTML tags
# =============================================================================

TAG_ROOT = "root"

TAG_H1 = "h1"
TAG_H2 = "h2"
TAG_H3 = "h3"
TAG_H4 = "h4"
TAG_H5 = "h5"
TAG_H6 = "h6"

TAG_UL = "ul"
TAG_OL = "ol"
TAG_LI = "li"

TAG_DL = "dl"
TAG_DT = "dt"
TAG_DD = "dd"

TAG_TABLE = "table"
TAG_TR = "tr"
TAG_TH = "th"
TAG_TD = "td"

TAG_P = "p"
TAG_DIV = "div"
TAG_BR = "br"

HTML_HEADER_TAGS = {
    TAG_H1,
    TAG_H2,
    TAG_H3,
    TAG_H4,
    TAG_H5,
    TAG_H6,
}

HTML_LIST_CONTAINER_TAGS = {
    TAG_UL,
    TAG_OL,
}

HTML_DEFINITION_LIST_CONTAINER_TAGS = {
    TAG_DL,
}

HTML_LIST_TAGS = {
    TAG_UL,
    TAG_OL,
    TAG_LI,
    TAG_DL,
    TAG_DT,
    TAG_DD,
}

HTML_TABLE_CELL_TAGS = {
    TAG_TH,
    TAG_TD,
}

HTML_TABLE_TAGS = {
    TAG_TABLE,
    TAG_TR,
    TAG_TH,
    TAG_TD,
}

HTML_BLOCK_TAGS = (
    {
        TAG_P,
        TAG_DIV,
        TAG_BR,
    }
    | HTML_HEADER_TAGS
    | HTML_LIST_TAGS
    | HTML_TABLE_TAGS
)

# =============================================================================
# Table Attributes
# =============================================================================

KEY_ROWSPAN = "rowspan"
KEY_NESTED_TABLES = "nested_tables"
KEY_ROWS = "rows"
KEY_MATRIX = "matrix"

ATTRIBUTE_COLSPAN = "colspan"
ATTRIBUTE_ROWSPAN = "rowspan"

DEFAULT_TABLE_SPAN = 1
METADATA_CLASS = "metadata"

# =============================================================================
# List Attributes
# =============================================================================
ATTRIBUTE_START = "start"
DEFAULT_ORDERED_LIST_START = 1

# =============================================================================
# Section filtering
# =============================================================================

SKIP_SECTIONS = {
    "see also",
    "references",
    "further reading",
    "bibliography",
    "external links",
    "notes",
    "footnotes",
    "citations",
    "sources",
    "works cited",
    "contents",
    "notes and references",
}


# =============================================================================
# Tagging
# =============================================================================
ROW_DOCUMENT_NORMALIZATION = "document_normalization"

DOCUMENT_NORMALIZATION_NAME = "htmlnorm"
DOCUMENT_NORMALIZATION_VERSION = "v1"
INFOBOX_MODE_KEEP = "infobox"
INFOBOX_MODE_DROP = "no_infobox"
TOKENIZER_MODE_NONE = "none"

# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--model-name", default="google-t5/t5-small")
    args = parser.parse_args()

    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    row = read_row(args.input, args.row)
    row = simplify_nq_example(row, tokenizer, True)

    print("===")
    print(row.get(ROW_DOCUMENT_NORMALIZATION))
    print(row.get(ROW_DOCUMENT_TITLE))
    print(html.unescape(row.get(ROW_DOCUMENT_URL)))
    print("===")
    print(row.get(ROW_DOCUMENT_TEXT))


def simplify_nq_example(row, tokenizer=None, remove_infobox=True):
    tag, document_title, document_text = extract_title_and_text(row, tokenizer, remove_infobox)

    return {
        ROW_EXAMPLE_ID: row.get(ROW_EXAMPLE_ID),
        ROW_DOCUMENT_URL: row.get(ROW_DOCUMENT_URL),
        ROW_QUESTION_TEXT: row.get(ROW_QUESTION_TEXT),
        ROW_DOCUMENT_TITLE: document_title,
        ROW_DOCUMENT_TEXT: document_text,
        ROW_DOCUMENT_NORMALIZATION: tag,
    }


def extract_text(row, tokenizer=None, remove_infobox=False):
    _, _, text = extract_title_and_text(row, tokenizer, remove_infobox)
    return text


def extract_title_and_text(row, tokenizer=None, remove_infobox=False):
    ast = parse_document_ast(row)

    merge_text_nodes_in_place(ast)
    clean_wikipedia_ui_text_nodes_in_place(ast)

    tag = htmlnorm_tag(tokenizer, remove_infobox)
    document_title = extract_h1_title(ast)

    ast = transform_headers_to_flat_sections(ast)
    clean_and_filter_sections_in_place(ast, SKIP_SECTIONS_NORMALIZED)
    transform_remove_metadata_tables_from_first_section_in_place(ast)
    transform_level_1_intro_sections_in_place(ast, remove_infobox=remove_infobox)
    ast = transform_tables_in_place(ast)
    ast = transform_lists_in_place(ast)

    text = flatten_text(ast)
    text = normalize_final_document_text(text)

    if tokenizer is not None:
        text = remove_intro_pronunciation_from_document_text(text, tokenizer)

    return tag, document_title, text


def extract_h1_title(node):
    if node[KEY_KIND] == KIND_ELEMENT and node.get(KEY_TAG) == TAG_H1:
        return node_text(node)

    if not has_children(node):
        return None

    for child in node[KEY_CHILDREN]:
        title = extract_h1_title(child)
        if title:
            return title

    return None


# =============================================================================
# Tagging
# =============================================================================


def htmlnorm_tag(tokenizer=None, remove_infobox=True):
    infobox_part = INFOBOX_MODE_DROP if remove_infobox else INFOBOX_MODE_KEEP
    tokenizer_part = tokenizer_tag(tokenizer)
    return f"{DOCUMENT_NORMALIZATION_NAME}-{DOCUMENT_NORMALIZATION_VERSION}-{infobox_part}-{tokenizer_part}"


def tokenizer_tag(tokenizer):
    if tokenizer is None:
        return TOKENIZER_MODE_NONE

    name = getattr(tokenizer, "name_or_path", None)
    if not name:
        return "tokenizer"

    return name.split("/")[-1].replace("-", "_")


# =============================================================================
# File reading
# =============================================================================


def open_jsonl(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")

    return open(path, encoding="utf-8")


def read_row(path, row_index):
    with open_jsonl(path) as file:
        for index, line in enumerate(file):
            if index == row_index:
                return json.loads(line)

    raise ValueError(f"Row {row_index} not found")


# =============================================================================
# HTML token utilities
# =============================================================================


def span(data, start_token, end_token):
    bytespan = data[start_token[TOKEN_END_BYTE] : end_token[TOKEN_START_BYTE]]
    text = bytespan.decode("utf8", errors="replace")
    return normalize_html_span(text)


def html_span(data, token):
    bytespan = data[token[TOKEN_START_BYTE] : token[TOKEN_END_BYTE]]
    text = bytespan.decode("utf8", errors="replace")
    return normalize_html_span(text)


def is_html_token(token):
    return token[TOKEN_HTML_TOKEN]


def is_closing_tag(token):
    return token[TOKEN_TEXT].startswith("</")


def is_opening_tag(token):
    return token[TOKEN_TEXT].startswith("<") and not is_closing_tag(token)


def tag_name(html_token):
    token_text = html_token[TOKEN_TEXT]
    token_text = token_text.strip().lower()

    token_text = token_text.lstrip("<").rstrip(">").strip()
    token_text = token_text.split(maxsplit=1)[0]

    if token_text.startswith("/"):
        return token_text[1:]

    return token_text


def html_has_visible_whitespace(text):
    stripped = TAG_RE.sub(SEPARATOR_EMPTY, text)
    return any(character.isspace() for character in stripped)


def parse_html_attributes(raw_html):
    attributes = {}

    if not raw_html:
        return attributes

    for key, value in ATTRIBUTE_RE.findall(raw_html):
        attributes[key.lower()] = value

    return attributes


def node_attributes(node):
    attributes = node.get(KEY_ATTRS_PARSED)

    if attributes is None:
        attributes = parse_html_attributes(node.get(KEY_ATTRS_HTML))
        node[KEY_ATTRS_PARSED] = attributes

    return attributes


def get_int_attribute(node, name, default=DEFAULT_TABLE_SPAN):
    try:
        return int(node_attributes(node).get(name, default))
    except ValueError:
        return default


# =============================================================================
# Byte-gap separator inference
# =============================================================================


def infer_separator(gap):
    if not gap:
        return SEPARATOR_EMPTY

    if gap == " ":
        return SEPARATOR_SPACE

    if "<" in gap:
        for match in TAG_NAME_RE.finditer(gap):
            if match.group(1).lower() in HTML_BLOCK_TAGS:
                return SEPARATOR_LINE_BREAK

        if html_has_visible_whitespace(gap):
            return SEPARATOR_SPACE

    return SEPARATOR_EMPTY


# =============================================================================
# AST node creation
# =============================================================================


def create_element_node(tag, token, attrs_html):
    node = {
        KEY_KIND: KIND_ELEMENT,
        KEY_TAG: tag,
        KEY_CHILDREN: [],
    }

    if token is not None:
        node[KEY_TOKEN] = token

    if attrs_html is not None:
        node[KEY_ATTRS_HTML] = attrs_html

    return node


def create_root_node():
    return {
        KEY_KIND: KIND_ROOT,
        KEY_TAG: TAG_ROOT,
        KEY_CHILDREN: [],
    }


def create_text_node(text, token=None):
    node = {
        KEY_KIND: KIND_TEXT,
        KEY_TEXT: text,
    }

    if token is not None:
        node[KEY_TOKEN] = token

    return node


def create_section_node(title=None, level=0):
    return {
        KEY_KIND: KIND_SECTION,
        KEY_TITLE: title if title else None,
        KEY_LEVEL: level,
        KEY_CHILDREN: [],
    }


# =============================================================================
# AST parsing
# =============================================================================


def parse_document_ast(row):
    html_bytes = row[ROW_DOCUMENT_HTML].encode("utf-8")
    document_tokens = row[ROW_DOCUMENT_TOKENS]

    root = create_root_node()
    stack = [root]

    previous_text_token = None

    for document_token in document_tokens:
        if is_html_token(document_token):
            current_tag_name = tag_name(document_token)

            if is_closing_tag(document_token):
                if len(stack) > 1 and stack[-1].get(KEY_TAG) == current_tag_name:
                    previous_text_token = None
                    stack.pop()
                continue

            raw_html = html_span(html_bytes, document_token)
            node = create_element_node(current_tag_name, document_token, raw_html)

            stack[-1][KEY_CHILDREN].append(node)
            stack.append(node)

            previous_text_token = None
            continue

        separator = SEPARATOR_EMPTY

        if previous_text_token is not None:
            gap = span(html_bytes, previous_text_token, document_token)
            separator = infer_separator(gap)

        if separator:
            stack[-1][KEY_CHILDREN].append(create_text_node(separator))

        token_text = html_span(html_bytes, document_token)
        token_text = normalize_token_text(token_text)
        text_node = create_text_node(token_text, token=document_token)
        stack[-1][KEY_CHILDREN].append(text_node)

        previous_text_token = document_token

    return root


# =============================================================================
# AST printing / debugging
# =============================================================================


def print_ast(node, indent=0, show_attrs=False):
    prefix = INDENT_UNIT * indent
    kind = node[KEY_KIND]

    if kind == KIND_TEXT:
        text = node[KEY_TEXT].replace(SEPARATOR_LINE_BREAK, "\\n")
        print(f"{prefix}{text!r}")
        return

    if kind == KIND_SECTION:
        title = node.get(KEY_TITLE)
        level = node.get(KEY_LEVEL)
        print(f"{prefix}SECTION(level={level}, title={title!r})")
        print_ast_children(node, indent)
        return

    if kind == KIND_ROOT:
        print(f"{prefix}<root>")
        print_ast_children(node, indent)
        print(f"{prefix}</root>")
        return

    if kind == KIND_ELEMENT:
        close_tag = f"</{node[KEY_TAG]}>"
        open_tag = node[KEY_ATTRS_HTML].lower() if show_attrs else f"<{node[KEY_TAG]}>"
        print(f"{prefix}{open_tag}")
        print_ast_children(node, indent)
        print(f"{prefix}{close_tag}")
        return

    raise ValueError(f"Unknown node kind: {kind}")


def print_ast_children(node, indent):
    for child in node[KEY_CHILDREN]:
        print_ast(child, indent + 1)


# =============================================================================
# Text normalization / cleaning
# =============================================================================


def normalize_unicode_text(text):
    """Normalize Unicode characters and remove unwanted control characters."""
    text = unicodedata.normalize("NFKC", text)

    result = []
    for ch in text:
        category = unicodedata.category(ch)

        if category == "Zs":
            result.append(SEPARATOR_SPACE)
        elif category.startswith("C") and ch != SEPARATOR_LINE_BREAK:
            continue
        else:
            result.append(ch)

    return SEPARATOR_EMPTY.join(result)


def normalize_html_span(text):
    """Normalize raw decoded HTML spans before AST insertion or gap analysis."""
    return normalize_unicode_text(text)


def normalize_token_text(text):
    """Normalize visible token text extracted from HTML."""
    text = normalize_unicode_text(text)

    if "<" in text:
        text = TAG_RE.sub(SEPARATOR_EMPTY, text)

    if "&" in text:
        text = html.unescape(text)
        text = normalize_unicode_text(text)

    return text


def normalize_inline_text(text):
    text = normalize_unicode_text(text)
    text = WHITESPACE_RE.sub(SEPARATOR_SPACE, text)
    return text.strip()


def normalize_block_text(text):
    text = normalize_unicode_text(text)
    text = MANY_NEWLINES_RE.sub(SEPARATOR_SECTION_BREAK, text)
    return text.strip()


def normalize_final_document_text(text):
    """Final htmlnorm pass after all structure rendering."""
    text = normalize_unicode_text(text)
    text = MANY_NEWLINES_RE.sub(SEPARATOR_SECTION_BREAK, text)
    return text


SKIP_SECTIONS_NORMALIZED = frozenset(title.lower() for title in SKIP_SECTIONS)

# =============================================================================
# AST text helpers
# =============================================================================


def merge_text_nodes_in_place(node):
    if node[KEY_KIND] == KIND_TEXT:
        return node

    new_children = []

    for child in node[KEY_CHILDREN]:
        transformed_child = merge_text_nodes_in_place(child)

        if new_children and new_children[-1][KEY_KIND] == KIND_TEXT and transformed_child[KEY_KIND] == KIND_TEXT:
            new_children[-1][KEY_TEXT] += transformed_child[KEY_TEXT]
        else:
            new_children.append(transformed_child)

    node[KEY_CHILDREN] = new_children
    return node


def append_node_text(node, parts):
    if node[KEY_KIND] == KIND_TEXT:
        parts.append(node[KEY_TEXT])
        return

    for child in node[KEY_CHILDREN]:
        append_node_text(child, parts)


def node_text(node):
    if node[KEY_KIND] == KIND_TEXT:
        return node[KEY_TEXT]

    parts = []
    append_node_text(node, parts)
    return SEPARATOR_EMPTY.join(parts)


# =============================================================================
# AST traversal helpers
# =============================================================================
def node_has_content(node):
    return any(child[KEY_KIND] != KIND_TEXT or child[KEY_TEXT].strip() for child in node[KEY_CHILDREN])


def has_children(node):
    return KEY_CHILDREN in node


def map_children_in_place(node, transform):
    if not has_children(node):
        return node

    node[KEY_CHILDREN] = [transform(child) for child in node[KEY_CHILDREN]]
    return node


def filter_map_children_in_place(node, transform):
    if not has_children(node):
        return node

    new_children = []

    for child in node[KEY_CHILDREN]:
        transformed = transform(child)
        if transformed is not None:
            new_children.append(transformed)

    node[KEY_CHILDREN] = new_children
    return node


# =============================================================================
# AST flattening / final rendering
# =============================================================================


def render_separator_between_children(previous_child, current_child):
    if previous_child is None:
        return SEPARATOR_EMPTY

    if previous_child[KEY_KIND] == KIND_SECTION or current_child[KEY_KIND] == KIND_SECTION:
        return SEPARATOR_SECTION_BREAK

    if current_child[KEY_KIND] == KIND_ELEMENT and current_child.get(KEY_TAG) in HTML_BLOCK_TAGS:
        return SEPARATOR_LINE_BREAK

    if previous_child[KEY_KIND] == KIND_ELEMENT and previous_child.get(KEY_TAG) in HTML_BLOCK_TAGS:
        return SEPARATOR_LINE_BREAK

    return SEPARATOR_EMPTY


def flatten_children(children, include_section_titles=True):
    parts = []
    previous_child = None

    for child in children:
        text = flatten_text(
            child,
            normalize=False,
            include_section_titles=include_section_titles,
        )

        if not text:
            continue

        separator = render_separator_between_children(previous_child, child)
        if separator:
            parts.append(separator)

        parts.append(text)
        previous_child = child

    return SEPARATOR_EMPTY.join(parts)


def flatten_text(node, normalize=True, include_section_titles=True):
    kind = node[KEY_KIND]

    if kind == KIND_TEXT:
        text = node[KEY_TEXT]

    elif kind == KIND_SECTION:
        body = flatten_children(
            node[KEY_CHILDREN],
            include_section_titles=include_section_titles,
        )

        title = node.get(KEY_TITLE)
        if include_section_titles and title:
            text = f"{title}{SEPARATOR_LINE_BREAK}{body}" if body else title
        else:
            text = body

    else:
        text = flatten_children(
            node[KEY_CHILDREN],
            include_section_titles=include_section_titles,
        )

    if normalize:
        text = text.strip()

    return text


# =============================================================================
# Section transforms
# =============================================================================


def heading_level(tag):
    if tag in HTML_HEADER_TAGS:
        return int(tag[1])

    return None


def transform_headers_to_flat_sections(root):
    new_root = create_root_node()
    current_section = create_section_node(title=None, level=0)

    def flush_current_section():
        if node_has_content(current_section):
            new_root[KEY_CHILDREN].append(current_section)

    def visit_top_level_node(node):
        nonlocal current_section

        if node[KEY_KIND] == KIND_ELEMENT and node.get(KEY_TAG) in HTML_HEADER_TAGS:
            flush_current_section()

            current_section = create_section_node(
                title=node_text(node),
                level=heading_level(node[KEY_TAG]),
            )
            return

        current_section[KEY_CHILDREN].append(node)

    for node in root[KEY_CHILDREN]:
        if node[KEY_KIND] == KIND_ELEMENT and node.get(KEY_TAG) not in HTML_HEADER_TAGS:
            has_direct_heading_child = any(
                child[KEY_KIND] == KIND_ELEMENT and child.get(KEY_TAG) in HTML_HEADER_TAGS
                for child in node[KEY_CHILDREN]
            )

            if has_direct_heading_child:
                for child in node[KEY_CHILDREN]:
                    visit_top_level_node(child)
            else:
                visit_top_level_node(node)
        else:
            visit_top_level_node(node)

    flush_current_section()

    return new_root


def clean_and_filter_sections_in_place(node, titles_to_remove):
    if node[KEY_KIND] == KIND_TEXT:
        return node

    def transform_child(child):
        if child[KEY_KIND] == KIND_SECTION:
            title = child.get(KEY_TITLE)
            if title is None:
                return None

            if title.lower() in titles_to_remove:
                return None

        return clean_and_filter_sections_in_place(child, titles_to_remove)

    return filter_map_children_in_place(node, transform_child)


# =============================================================================
# List transforms
# =============================================================================


def list_node_to_text(node, indent=0):
    tag = node.get(KEY_TAG)

    if tag == TAG_OL:
        return ordered_list_to_text(node, indent=indent)

    if tag == TAG_UL:
        return unordered_list_to_text(node, indent=indent)

    if tag == TAG_DL:
        return description_list_to_text(node, indent=indent)

    return SEPARATOR_EMPTY


def transform_lists_in_place(node, indent=0):
    if node[KEY_KIND] == KIND_TEXT:
        return node

    tag = node.get(KEY_TAG)

    if tag in HTML_LIST_CONTAINER_TAGS or tag in HTML_DEFINITION_LIST_CONTAINER_TAGS:
        text = list_node_to_text(node, indent=indent)
        if not text.strip():
            return create_text_node(SEPARATOR_EMPTY)

        return create_text_node(SEPARATOR_LINE_BREAK + text.rstrip() + SEPARATOR_LINE_BREAK)

    return map_children_in_place(
        node,
        lambda child: transform_lists_in_place(child, indent=indent),
    )


def list_item_children_to_text_and_nested_lists(item_node):
    text_parts = []
    nested_lists = []

    for child in item_node[KEY_CHILDREN]:
        if child[KEY_KIND] == KIND_ELEMENT and (
            child.get(KEY_TAG) in HTML_LIST_CONTAINER_TAGS or child.get(KEY_TAG) in HTML_DEFINITION_LIST_CONTAINER_TAGS
        ):
            nested_lists.append(child)
        else:
            text_parts.append(node_text(child))

    return SEPARATOR_EMPTY.join(text_parts), nested_lists


def render_nested_lists(nested_lists, indent):
    lines = []

    for nested_list in nested_lists:
        nested_text = list_node_to_text(nested_list, indent=indent + 1)
        if nested_text.strip():
            lines.append(nested_text.rstrip())

    return lines


def ordered_list_to_text(node, indent=0):
    lines = []
    prefix = INDENT_UNIT * indent
    item_index = get_int_attribute(
        node,
        ATTRIBUTE_START,
        default=DEFAULT_ORDERED_LIST_START,
    )

    for child in node[KEY_CHILDREN]:
        if child[KEY_KIND] != KIND_ELEMENT or child.get(KEY_TAG) != TAG_LI:
            continue

        item_text, nested_lists = list_item_children_to_text_and_nested_lists(child)

        marker = f"{item_index}."
        item_index += 1

        if item_text:
            lines.append(f"{prefix}{marker} {item_text}")
        else:
            lines.append(f"{prefix}{marker}")

        lines.extend(render_nested_lists(nested_lists, indent))

    return SEPARATOR_LINE_BREAK.join(lines) + SEPARATOR_LINE_BREAK


def unordered_list_to_text(node, indent=0):
    lines = []
    prefix = INDENT_UNIT * indent

    for child in node[KEY_CHILDREN]:
        if child[KEY_KIND] != KIND_ELEMENT or child.get(KEY_TAG) != TAG_LI:
            continue

        item_text, nested_lists = list_item_children_to_text_and_nested_lists(child)

        if item_text:
            lines.append(f"{prefix}{UNORDERED_LIST_MARKER} {item_text}")
        else:
            lines.append(f"{prefix}{UNORDERED_LIST_MARKER}")

        lines.extend(render_nested_lists(nested_lists, indent))

    return SEPARATOR_LINE_BREAK.join(lines) + SEPARATOR_LINE_BREAK


def description_list_to_text(node, indent=0):
    lines = []
    prefix = INDENT_UNIT * indent
    pending_terms = []

    for child in node[KEY_CHILDREN]:
        if child[KEY_KIND] != KIND_ELEMENT:
            continue

        tag = child.get(KEY_TAG)

        if tag == TAG_DT:
            term_text, nested_lists = description_child_to_text_and_nested_lists(child)
            if term_text:
                pending_terms.append(term_text)

            # Rare, but preserve nested content inside dt.
            lines.extend(render_nested_lists(nested_lists, indent))
            continue

        if tag == TAG_DD:
            definition_text, nested_lists = description_child_to_text_and_nested_lists(child)

            if pending_terms:
                for term in pending_terms:
                    if definition_text:
                        lines.append(f"{prefix}{term}: {definition_text}")
                    else:
                        lines.append(f"{prefix}{term}:")
                pending_terms = []
            elif definition_text:
                # No dt before this dd: render as standalone line.
                lines.append(f"{prefix}{definition_text}")

            lines.extend(render_nested_lists(nested_lists, indent))
            continue

        # Fallback: preserve unexpected content as its own line.
        text = node_text(child)
        if text:
            lines.append(f"{prefix}{text}")

    for term in pending_terms:
        lines.append(f"{prefix}{term}")

    return SEPARATOR_LINE_BREAK.join(lines) + SEPARATOR_LINE_BREAK


def description_child_to_text_and_nested_lists(node):
    text_parts = []
    nested_lists = []

    for child in node[KEY_CHILDREN]:
        if child[KEY_KIND] == KIND_ELEMENT and (
            child.get(KEY_TAG) in HTML_LIST_CONTAINER_TAGS or child.get(KEY_TAG) in HTML_DEFINITION_LIST_CONTAINER_TAGS
        ):
            nested_lists.append(child)
        else:
            text_parts.append(node_text(child))

    return SEPARATOR_EMPTY.join(text_parts), nested_lists


# =============================================================================
# Table parsing
# =============================================================================


def parse_table(table_node):
    rows = parse_table_rows(table_node)
    matrix = expand_table_spans(rows)

    return {
        KEY_ROWS: rows,
        KEY_MATRIX: matrix,
    }


def parse_table_rows(table_node):
    rows = []

    for child in table_node[KEY_CHILDREN]:
        if child[KEY_KIND] != KIND_ELEMENT:
            continue

        if child.get(KEY_TAG) != TAG_TR:
            continue

        rows.append(parse_table_row(child))

    return rows


def parse_table_row(row_node):
    cells = []

    for child in row_node[KEY_CHILDREN]:
        if child[KEY_KIND] != KIND_ELEMENT:
            continue

        if child.get(KEY_TAG) not in HTML_TABLE_CELL_TAGS:
            continue

        cells.append(parse_table_cell(child))

    return cells


def parse_table_cell(cell_node):
    nested_tables = find_direct_nested_tables(cell_node)

    text = table_cell_text_without_nested_tables(cell_node)

    return {
        KEY_TAG: cell_node.get(KEY_TAG),
        KEY_TEXT: normalize_block_text(text),
        KEY_NODE: cell_node,
        KEY_COLSPAN: max(1, get_int_attribute(cell_node, ATTRIBUTE_COLSPAN)),
        KEY_ROWSPAN: max(1, get_int_attribute(cell_node, ATTRIBUTE_ROWSPAN)),
        KEY_NESTED_TABLES: [parse_table(table) for table in nested_tables],
    }


def find_direct_nested_tables(node):
    nested_tables = []

    for child in node[KEY_CHILDREN]:
        if child[KEY_KIND] == KIND_ELEMENT and child.get(KEY_TAG) == TAG_TABLE:
            nested_tables.append(child)

    return nested_tables


def table_cell_text_without_nested_tables(cell_node):
    parts = []

    for child in cell_node[KEY_CHILDREN]:
        if child[KEY_KIND] == KIND_TEXT:
            text = normalize_inline_text(child[KEY_TEXT])
            if text:
                parts.append(text)
            continue

        if child[KEY_KIND] != KIND_ELEMENT:
            continue

        tag = child.get(KEY_TAG)

        if tag == TAG_TABLE:
            continue

        if tag in HTML_LIST_CONTAINER_TAGS or tag in HTML_DEFINITION_LIST_CONTAINER_TAGS:
            text = normalize_block_text(list_node_to_text(child))
        else:
            text = normalize_inline_text(node_text(child))

        if text:
            parts.append(text)

    return SEPARATOR_LINE_BREAK.join(parts)


def expand_table_spans(rows):
    matrix = []
    active_rowspans = {}

    for row in rows:
        matrix_row = []
        column_index = 0

        def consume_active_rowspans(matrix_row):
            nonlocal column_index

            while column_index in active_rowspans:
                cell, remaining = active_rowspans[column_index]
                matrix_row.append(cell)

                remaining -= 1
                if remaining <= 0:
                    del active_rowspans[column_index]
                else:
                    active_rowspans[column_index] = (cell, remaining)

                column_index += 1

        consume_active_rowspans(matrix_row)

        for cell in row:
            consume_active_rowspans(matrix_row)

            colspan = cell.get(KEY_COLSPAN, 1)
            rowspan = cell.get(KEY_ROWSPAN, 1)

            for _ in range(colspan):
                matrix_row.append(cell)

                if rowspan > 1:
                    active_rowspans[column_index] = (cell, rowspan - 1)

                column_index += 1

        consume_active_rowspans(matrix_row)

        if matrix_row:
            matrix.append(matrix_row)

    # Flush trailing rowspans if malformed table ends early.
    while active_rowspans:
        matrix_row = []
        column_index = 0

        while active_rowspans:
            if column_index in active_rowspans:
                cell, remaining = active_rowspans[column_index]
                matrix_row.append(cell)

                remaining -= 1
                if remaining <= 0:
                    del active_rowspans[column_index]
                else:
                    active_rowspans[column_index] = (cell, remaining)
            else:
                matrix_row.append(None)

            column_index += 1

            if column_index > max(active_rowspans.keys(), default=-1) + 1:
                break

        matrix.append(matrix_row)

    return matrix


# =============================================================================
# Table rendering
# =============================================================================
TABLE_TYPE_EMPTY = "empty"
TABLE_TYPE_NESTED = "nested"
TABLE_TYPE_KEY_VALUE = "key_value"
TABLE_TYPE_MATRIX = "matrix"
TABLE_TYPE_ROW_WISE = "row_wise"
TABLE_TYPE_SINGLE_COLUMN = "single_column"


def transform_tables_in_place(node):
    if node[KEY_KIND] == KIND_TEXT:
        return node

    if node.get(KEY_TAG) == TAG_TABLE:
        table = parse_table(node)
        text = render_table(table)

        if not text.strip():
            return create_text_node(SEPARATOR_EMPTY)

        return create_text_node(SEPARATOR_LINE_BREAK + text.strip() + SEPARATOR_LINE_BREAK)

    return map_children_in_place(node, transform_tables_in_place)


def render_table(table):
    table_type = classify_table(table)

    if table_type == TABLE_TYPE_EMPTY:
        return SEPARATOR_EMPTY

    if table_type == TABLE_TYPE_NESTED:
        return render_nested_table(table)

    if table_type == TABLE_TYPE_KEY_VALUE:
        return render_key_value_table(table)

    if table_type == TABLE_TYPE_SINGLE_COLUMN:
        return render_single_column_table(table)

    if table_type == TABLE_TYPE_MATRIX:
        return render_matrix_table(table)

    return render_row_wise_table(table)


def classify_table(table):
    matrix = table.get(KEY_MATRIX, [])

    if not matrix:
        return TABLE_TYPE_EMPTY

    if table_looks_like_key_value_table_model(table):
        return TABLE_TYPE_KEY_VALUE

    max_columns = max(len(row) for row in matrix)

    if max_columns <= 1:
        return TABLE_TYPE_SINGLE_COLUMN

    if table_has_meaningful_header_row(table):
        return TABLE_TYPE_ROW_WISE

    if table_has_nested_tables(table):
        return TABLE_TYPE_NESTED

    return TABLE_TYPE_MATRIX


def table_has_nested_tables(table):
    for row in table.get(KEY_ROWS, []):
        for cell in row:
            if cell.get(KEY_NESTED_TABLES):
                return True
    return False


def render_cell_value(cell):
    if cell is None:
        return SEPARATOR_EMPTY

    parts = []

    text = cell_text(cell)
    if text:
        parts.append(text)

    nested_text = render_cell_nested_tables(cell)
    if nested_text:
        parts.append(nested_text)

    return SEPARATOR_LINE_BREAK.join(parts)


def cell_text(cell):
    if cell is None:
        return SEPARATOR_EMPTY
    return normalize_block_text(cell.get(KEY_TEXT, SEPARATOR_EMPTY))


def cell_is_header(cell):
    return cell is not None and cell.get(KEY_TAG) == TAG_TH


def cell_has_content(cell):
    if cell is None:
        return False

    return bool(cell_text(cell) or cell.get(KEY_NESTED_TABLES))


def non_empty_cells(row):
    return [cell for cell in row if cell_has_content(cell)]


def table_looks_like_key_value_table_model(table):
    rows = table.get(KEY_ROWS, [])

    meaningful_rows = 0
    key_value_rows = 0

    for row in rows:
        cells = non_empty_cells(row)
        if not cells:
            continue

        meaningful_rows += 1

        if len(cells) == 1:
            key_value_rows += 1
            continue

        first = cells[0]
        second = cells[1]

        if cell_is_header(first) and render_cell_value(second):
            key_value_rows += 1
            continue

        if len(cells) == 2 and looks_like_label(cell_text(first)):
            key_value_rows += 1
            continue

    return meaningful_rows > 0 and key_value_rows / meaningful_rows >= 0.5


def looks_like_label(text):
    if not text:
        return False

    if len(text) > 80:
        return False

    if text.endswith(":"):
        return True

    # Short labels like "Born", "Occupation", "Capital"
    return len(text.split()) <= 6


def table_has_meaningful_header_row(table):
    matrix = table.get(KEY_MATRIX, [])
    if not matrix:
        return False

    first_row = matrix[0]
    cells = non_empty_cells(first_row)

    if not cells:
        return False

    header_count = sum(1 for cell in cells if cell_is_header(cell))

    return header_count >= max(1, len(cells) // 2)


# =============================================================================
# Table renderers
# =============================================================================
def render_key_value_table(table):
    lines = []
    pending_key = None

    for row in table.get(KEY_ROWS, []):
        cells = non_empty_cells(row)
        if not cells:
            continue

        if len(cells) == 1:
            only = cells[0]
            text = cell_text(only)
            nested_text = render_cell_nested_tables(only)

            if cell_is_header(only):
                if text:
                    pending_key = text.rstrip(":")
                continue

            value = join_non_empty([text, nested_text], separator=SEPARATOR_LINE_BREAK)

            if pending_key and value:
                lines.append(f"{pending_key}: {value}")
                pending_key = None
            elif value:
                lines.append(value)

            continue

        key = cell_text(cells[0]).rstrip(":")
        values = [render_cell_value(cell) for cell in cells[1:] if render_cell_value(cell)]

        if key and values:
            lines.append(f"{key}: {TABLE_CELL_SEPARATOR.join(values)}")
            pending_key = None
        elif key:
            pending_key = key
        elif values:
            value = TABLE_CELL_SEPARATOR.join(values)
            if pending_key:
                lines.append(f"{pending_key}: {value}")
                pending_key = None
            else:
                lines.append(value)

    return SEPARATOR_LINE_BREAK.join(lines)


def render_cell_nested_tables(cell):
    nested_tables = cell.get(KEY_NESTED_TABLES, [])
    rendered = []

    for nested_table in nested_tables:
        text = render_table(nested_table)
        if text:
            rendered.append(text)

    return SEPARATOR_LINE_BREAK.join(rendered)


def join_non_empty(parts, separator):
    parts = [part for part in parts if part]
    return separator.join(parts)


def render_row_wise_table(table):
    matrix = table.get(KEY_MATRIX, [])
    if not matrix:
        return SEPARATOR_EMPTY

    header = [cell_text(cell) for cell in matrix[0]]
    body = matrix[1:]

    lines = []

    for row in body:
        pairs = []

        for column_name, cell in zip(header, row):
            value = render_cell_value(cell)

            if not value:
                continue

            if column_name:
                pairs.append(f"{column_name}: {value}")
            else:
                pairs.append(value)

        if pairs:
            lines.append(TABLE_CELL_SEPARATOR.join(pairs))

    return SEPARATOR_LINE_BREAK.join(lines)


def render_matrix_table(table):
    lines = []

    for row in table.get(KEY_MATRIX, []):
        cells = [render_cell_value(cell) for cell in row]
        cells = [cell for cell in cells if cell]

        if cells:
            lines.append(TABLE_CELL_SEPARATOR.join(cells))

    return SEPARATOR_LINE_BREAK.join(lines)


def render_single_column_table(table):
    lines = []

    for row in table.get(KEY_MATRIX, []):
        cells = [render_cell_value(cell) for cell in row if render_cell_value(cell)]

        if cells:
            lines.append(cells[0])

    return SEPARATOR_LINE_BREAK.join(lines)


def render_nested_table(table):
    lines = []

    for row in table.get(KEY_ROWS, []):
        row_parts = []

        for cell in row:
            text = cell_text(cell)
            if text:
                row_parts.append(text)

            for nested_table in cell.get(KEY_NESTED_TABLES, []):
                nested_text = render_table(nested_table)
                if nested_text:
                    row_parts.append(nested_text)

        if row_parts:
            lines.append(SEPARATOR_LINE_BREAK.join(row_parts))

    return SEPARATOR_SECTION_BREAK.join(lines)


# =============================================================================
# Level-1 intro transforms
# =============================================================================


def transform_level_1_intro_sections_in_place(
    node,
    move_tables_after_paragraphs=True,
    remove_infobox=False,
):
    if node[KEY_KIND] == KIND_TEXT:
        return node

    if node[KEY_KIND] == KIND_SECTION and node.get(KEY_LEVEL) == 1:
        node[KEY_CHILDREN] = transform_level_1_intro_children(
            node[KEY_CHILDREN],
            move_tables_after_paragraphs=move_tables_after_paragraphs,
            remove_infobox=remove_infobox,
        )
        return node

    return map_children_in_place(
        node,
        lambda child: transform_level_1_intro_sections_in_place(
            child,
            move_tables_after_paragraphs=move_tables_after_paragraphs,
            remove_infobox=remove_infobox,
        ),
    )


def transform_level_1_intro_children(
    children,
    move_tables_after_paragraphs=True,
    remove_infobox=False,
):
    top_level_paragraphs = []
    top_level_tables = []
    other_top_level_blocks = []

    for child in children:
        if child[KEY_KIND] != KIND_ELEMENT:
            # Skip loose text like "Jump to: navigation, search".
            continue

        tag = child.get(KEY_TAG)

        if tag == TAG_P:
            if node_text(child).strip():
                top_level_paragraphs.append(child)
            continue

        if tag == TAG_TABLE:
            if not remove_infobox:
                top_level_tables.append(child)
            continue

        other_top_level_blocks.append(child)

    if remove_infobox:
        return top_level_paragraphs + other_top_level_blocks

    if move_tables_after_paragraphs:
        return top_level_paragraphs + top_level_tables + other_top_level_blocks

    return top_level_paragraphs + other_top_level_blocks + top_level_tables


# =============================================================================
# Level 1 metadata tables cleanup
# =============================================================================
def transform_remove_metadata_tables_from_first_section_in_place(root):
    first_section = first_child_section(root)
    if first_section is None:
        return root

    first_section[KEY_CHILDREN] = [child for child in first_section[KEY_CHILDREN] if not is_metadata_table(child)]

    return root


def first_child_section(node):
    for child in node.get(KEY_CHILDREN, []):
        if child[KEY_KIND] == KIND_SECTION:
            return child

    return None


def is_metadata_table(node):
    if node[KEY_KIND] != KIND_ELEMENT:
        return False

    if node.get(KEY_TAG) != TAG_TABLE:
        return False

    class_attr = node_attributes(node).get("class", "")
    classes = set(class_attr.lower().split())

    return METADATA_CLASS in classes


# =============================================================================
# Wikipedia UI cleanup
# =============================================================================


def clean_wikipedia_ui_text(text):
    # Wikipedia may include UI artifacts such as "[edit]" from templates.
    # These are removed as part of normalization.
    return WIKIPEDIA_UI_TEXT_RE.sub(SEPARATOR_EMPTY, text)


def clean_wikipedia_ui_text_nodes_in_place(node):
    if node[KEY_KIND] == KIND_TEXT:
        node[KEY_TEXT] = clean_wikipedia_ui_text(node[KEY_TEXT])
        return node

    return map_children_in_place(node, clean_wikipedia_ui_text_nodes_in_place)


# =============================================================================
# Wikipedia intro pronunciation cleanup for <unk>
# =============================================================================


def remove_intro_pronunciation_from_document_text(text, tokenizer, max_chars=DEFAULT_INTRO_PRONUNCIATION_MAX_CHARS):
    prefix = text[:max_chars]
    suffix = text[max_chars:]

    cleaned_prefix = remove_intro_pronunciation_parentheses(prefix, tokenizer)

    return cleaned_prefix + suffix


def remove_intro_pronunciation_parentheses(text, tokenizer):
    spans_to_remove = []

    for start, end, candidate in iter_parenthesized_blocks(text):
        if contains_unknown_token(candidate, tokenizer):
            spans_to_remove.append((include_preceding_space(text, start), end))

    # Pronunciation parentheticals in Wikipedia leads often contain IPA or other
    # symbols that T5 maps to <unk>. We scan every parenthesized block in the
    # intro prefix rather than stopping at the first one: the first block can be
    # title metadata such as "(film)", and some leads contain several
    # pronunciation blocks. Removing all <unk>-producing blocks keeps FirstP
    # inputs from spending early token budget on tokenizer artifacts.
    for start, end in reversed(spans_to_remove):
        text = text[:start] + text[end:]

    return text


def iter_parenthesized_blocks(text):
    start = None
    depth = 0

    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == ")" and depth > 0:
            depth -= 1

            if depth == 0 and start is not None:
                yield start, index + 1, text[start : index + 1]
                start = None


def contains_unknown_token(text, tokenizer):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.unk_token_id in token_ids


def include_preceding_space(text, start):
    if start > 0 and text[start - 1] == SEPARATOR_SPACE:
        return start - 1

    return start


if __name__ == "__main__":
    main()
