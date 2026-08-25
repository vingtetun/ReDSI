import html
from urllib.parse import parse_qs, unquote, urlsplit

KEY_PARAMETER_TITLE = "title"
H1_OPEN = "<H1>"
H1_CLOSE = "</H1>"


def from_url(url):
    url = html.unescape(url)
    parts = urlsplit(url)
    qs = parse_qs(parts.query)

    title = qs[KEY_PARAMETER_TITLE][0]
    title = unquote(title)
    return title.replace("_", " ")


def from_h1(text):
    h1_open = text.find(H1_OPEN)
    if h1_open == -1:
        return ""

    h1_close = text.find(H1_CLOSE, h1_open + 4)
    if h1_close == -1:
        return ""

    title = text[h1_open + 4 : h1_close]
    return title.strip()
