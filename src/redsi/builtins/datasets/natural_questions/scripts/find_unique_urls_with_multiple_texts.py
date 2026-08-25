#!/usr/bin/env python3
"""
Find URLs that map to multiple distinct document_text strings, and print a
human-friendly diff context around the FIRST differing character.
"""

import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from tqdm.auto import tqdm

from redsi.builtins.datasets.natural_questions import features
from redsi.builtins.datasets.natural_questions.loader import load

SPLITS = ["train", "validation"]


############################################################################################
# Utils
############################################################################################
def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def first_diff_index(a, b):
    """Return index of first differing character, or None if identical."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1

    if i == len(a) and i == len(b):
        return None

    # Prefix case (one string ended)
    if i == n:
        return n

    return i


def _escape_ws(text):
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def diff_window(a, b, idx, left, right, escape_ws=True):
    """
    Return a readable multi-line diff window with a caret at the first diff.

    Shows:
      - shared context on both sides (window)
      - a caret marker
      - the differing chars (or <EOF>)
    """
    a_start = max(0, idx - left)
    a_end = min(len(a), idx + right)
    b_start = max(0, idx - left)
    b_end = min(len(b), idx + right)

    a_snip = a[a_start:a_end]
    b_snip = b[b_start:b_end]

    if escape_ws:
        a_snip = _escape_ws(a_snip)
        b_snip = _escape_ws(b_snip)

    caret_pos = idx - a_start
    caret = (" " * max(0, caret_pos)) + "^"

    a_ch = a[idx] if idx < len(a) else "<EOF>"
    b_ch = b[idx] if idx < len(b) else "<EOF>"

    out = []
    out.append("A: " + a_snip)
    out.append("   " + caret)
    out.append("B: " + b_snip)
    out.append("   " + caret)
    out.append(f"Δ at char {idx}: {repr(a_ch)} → {repr(b_ch)}")
    return "\n".join(out)


def pick_two_texts(texts, mode):
    """
    Pick two representative strings among variants.

    mode:
      - first_two: first two distinct variants (stable)
      - shortest_longest: tends to amplify structural differences
    """
    if len(texts) < 2:
        return None, None

    if mode == "first_two":
        return texts[0], texts[1]

    # shortest_longest
    ordered = sorted(texts, key=len)
    return ordered[0], ordered[-1]


############################################################################################
# Script arguments
############################################################################################
@dataclass
class Params:
    data_source: str


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Dataset folder containing train/validation files")
    ap.add_argument("--max_urls", type=int, default=20, help="Max problematic URLs to print")
    ap.add_argument("--left", type=int, default=90, help="Chars to show on the left of first diff")
    ap.add_argument("--right", type=int, default=110, help="Chars to show on the right of first diff")
    ap.add_argument(
        "--pair",
        choices=["first_two", "shortest_longest"],
        default="shortest_longest",
        help="Which two variants to compare per URL (default amplifies differences)",
    )
    ap.add_argument(
        "--no_escape_ws",
        action="store_true",
        help="Do not escape newlines/tabs in snippets (less compact output)",
    )
    return ap.parse_args()


############################################################################################
# Main
############################################################################################
def main():
    args = parse_args()

    params = Params(args.source)
    ds = load(params, -1, None)

    # url -> {text_md5 -> representative text}
    url_to_variants = defaultdict(dict)

    total = sum(len(ds[s]) for s in SPLITS)
    with tqdm(total=total, desc="Scanning") as bar:
        for split in SPLITS:
            for ex in ds[split]:
                bar.update(1)

                url = ex[features.KEY_DOCUMENT_URL]
                if "oldid=" not in url:
                    raise RuntimeError(f"URL missing oldid: {url}")

                text = ex[features.KEY_DOCUMENT_TEXT]
                h = md5(text)

                # keep one representative per distinct text string
                if h not in url_to_variants[url]:
                    url_to_variants[url][h] = text

    # Sort URLs by number of variants (descending)
    problems = [(url, len(variants)) for url, variants in url_to_variants.items() if len(variants) > 1]
    problems.sort(key=lambda x: x[1], reverse=True)

    print()
    print(f"Found {len(problems)} URLs with multiple document_text variants.")

    if not problems:
        return

    print()

    for i, (url, nvar) in enumerate(problems[: args.max_urls], start=1):
        texts = list(url_to_variants[url].values())
        a, b = pick_two_texts(texts, args.pair)

        idx = first_diff_index(a, b)
        if idx is None:
            # Shouldn't happen since texts differ, but keep it robust
            continue

        print("=" * 96)
        print(f"[{i:02d}] {nvar} variants  |  first diff @ char {idx}  |  len(A)={len(a)} len(B)={len(b)}")
        print(url)
        print("-" * 96)
        print(
            diff_window(
                a,
                b,
                idx,
                left=args.left,
                right=args.right,
                escape_ws=not args.no_escape_ws,
            )
        )
        print()

    # Small hint for users
    if len(problems) > args.max_urls:
        print(f"(Showing first {args.max_urls} / {len(problems)} problematic URLs.)")


if __name__ == "__main__":
    main()
