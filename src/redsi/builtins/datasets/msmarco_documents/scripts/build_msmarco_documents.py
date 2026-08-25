#!/usr/bin/env python3

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

DEFAULT_DATA_DIR = "build/msmarco_documents/data/sources"
DEFAULT_OUTPUT_DIR = "build/msmarco_documents/data"


def smart_open(path, mode):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def is_empty_text(text):
    return not (text or "").strip()


def load_queries(path):
    queries = {}

    with smart_open(path, "rt") as fin:
        for line in tqdm(fin, desc=f"Loading queries {Path(path).name}"):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            qid, query = parts
            queries[qid] = query

    return queries


def load_qrels(path):
    qrels_by_qid = defaultdict(list)
    positive_docids = set()
    total_qrels = 0
    positive_qrels = 0

    with smart_open(path, "rt") as fin:
        for line in tqdm(fin, desc=f"Loading qrels {Path(path).name}"):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            qid, _, docid, relevance = parts[:4]
            relevance = int(relevance)

            total_qrels += 1

            if relevance <= 0:
                continue

            qrels_by_qid[qid].append(docid)
            positive_docids.add(docid)
            positive_qrels += 1

    return qrels_by_qid, positive_docids, total_qrels, positive_qrels


def filter_qrels_to_docids(qrels_by_qid, valid_docids):
    filtered = defaultdict(list)
    removed_rows = 0
    removed_queries = 0

    for qid, docids in qrels_by_qid.items():
        kept = sorted({docid for docid in docids if docid in valid_docids})
        removed_rows += len(set(docids)) - len(kept)

        if kept:
            filtered[qid].extend(kept)
        else:
            removed_queries += 1

    return filtered, {
        "qrel_rows_removed_by_doc_filter": removed_rows,
        "queries_removed_by_doc_filter": removed_queries,
    }


def write_query_split(output_path, queries, qrels_by_qid):
    rows_written = 0
    query_count = 0
    missing_queries = 0

    with smart_open(output_path, "wt") as fout:
        for qid in tqdm(sorted(qrels_by_qid), desc=f"Writing {output_path.name}"):
            query = queries.get(qid)

            if query is None:
                missing_queries += 1
                continue

            docids = sorted(set(qrels_by_qid[qid]))
            if not docids:
                continue

            query_count += 1

            for docid in docids:
                row = {
                    "kind": "query",
                    "query_id": qid,
                    "text": query,
                    "docid": docid,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    return {
        "queries_with_positive_qrels": query_count,
        "query_doc_rows": rows_written,
        "missing_queries": missing_queries,
    }


def write_active_documents(docs_path, output_path, keep_docids, drop_empty_body=True):
    written = 0
    scanned = 0
    malformed = 0
    skipped_empty_body = 0
    skipped_inactive = 0
    written_docids = set()
    missing_docids = set(keep_docids)

    with smart_open(docs_path, "rt") as fin, smart_open(output_path, "wt") as fout:
        for line in tqdm(fin, desc="Writing active documents"):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 3)
            if len(parts) != 4:
                malformed += 1
                continue

            docid, url, title, body = parts
            scanned += 1

            if docid not in keep_docids:
                skipped_inactive += 1
                continue

            if drop_empty_body and is_empty_text(body):
                print(
                    f"\nSkipped | "
                    f"docid={docid:<12} | "
                    f"title={title[:40]!r:<42} | "
                    f"url={url[:80]!r:<82} | "
                    f"body={body[:80]!r}"
                )
                skipped_empty_body += 1
                missing_docids.discard(docid)
                continue

            row = {
                "docid": docid,
                "url": url,
                "title": title,
                "body": body,
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            written_docids.add(docid)
            missing_docids.discard(docid)

    return {
        "official_docs_scanned": scanned,
        "documents_written": written,
        "malformed_doc_lines": malformed,
        "inactive_docs_skipped": skipped_inactive,
        "empty_body_docs_skipped": skipped_empty_body,
        "missing_qrel_docids": len(missing_docids),
    }, written_docids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--docs_file", default="msmarco-docs.tsv.gz")
    parser.add_argument("--train_queries_file", default="msmarco-doctrain-queries.tsv.gz")
    parser.add_argument("--dev_queries_file", default="msmarco-docdev-queries.tsv.gz")
    parser.add_argument("--train_qrels_file", default="msmarco-doctrain-qrels.tsv.gz")
    parser.add_argument("--dev_qrels_file", default="msmarco-docdev-qrels.tsv.gz")

    parser.add_argument(
        "--keep_empty_body",
        action="store_true",
        help="Do not drop documents whose body is empty or whitespace.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = data_dir / args.docs_file
    train_queries_path = data_dir / args.train_queries_file
    dev_queries_path = data_dir / args.dev_queries_file
    train_qrels_path = data_dir / args.train_qrels_file
    dev_qrels_path = data_dir / args.dev_qrels_file

    documents_output_path = output_dir / "documents.jsonl.gz"
    train_output_path = output_dir / "train_queries.jsonl.gz"
    validation_output_path = output_dir / "validation_queries.jsonl.gz"
    stats_output_path = output_dir / "stats.json"

    train_queries = load_queries(train_queries_path)
    dev_queries = load_queries(dev_queries_path)

    train_qrels, train_docids, train_total_qrels, train_positive_qrels = load_qrels(train_qrels_path)
    dev_qrels, dev_docids, dev_total_qrels, dev_positive_qrels = load_qrels(dev_qrels_path)

    active_docids = train_docids | dev_docids

    document_stats, valid_docids = write_active_documents(
        docs_path=docs_path,
        output_path=documents_output_path,
        keep_docids=active_docids,
        drop_empty_body=not args.keep_empty_body,
    )

    filtered_train_qrels, train_filter_stats = filter_qrels_to_docids(train_qrels, valid_docids)
    filtered_dev_qrels, dev_filter_stats = filter_qrels_to_docids(dev_qrels, valid_docids)

    train_stats = write_query_split(
        output_path=train_output_path,
        queries=train_queries,
        qrels_by_qid=filtered_train_qrels,
    )

    validation_stats = write_query_split(
        output_path=validation_output_path,
        queries=dev_queries,
        qrels_by_qid=filtered_dev_qrels,
    )

    stats = {
        "source_files": {
            "docs": str(docs_path),
            "train_queries": str(train_queries_path),
            "dev_queries": str(dev_queries_path),
            "train_qrels": str(train_qrels_path),
            "dev_qrels": str(dev_qrels_path),
        },
        "outputs": {
            "documents": str(documents_output_path),
            "train": str(train_output_path),
            "validation": str(validation_output_path),
        },
        "preprocessing": {
            "drop_empty_body": not args.keep_empty_body,
        },
        "queries": {
            "train_queries_loaded": len(train_queries),
            "dev_queries_loaded": len(dev_queries),
            "train_queries_with_positive_qrels": train_stats["queries_with_positive_qrels"],
            "dev_queries_with_positive_qrels": validation_stats["queries_with_positive_qrels"],
            "train_query_doc_rows": train_stats["query_doc_rows"],
            "dev_query_doc_rows": validation_stats["query_doc_rows"],
            "train_missing_queries": train_stats["missing_queries"],
            "dev_missing_queries": validation_stats["missing_queries"],
            "train_queries_removed_by_doc_filter": train_filter_stats["queries_removed_by_doc_filter"],
            "dev_queries_removed_by_doc_filter": dev_filter_stats["queries_removed_by_doc_filter"],
        },
        "qrels": {
            "train_total_qrels": train_total_qrels,
            "train_positive_qrels": train_positive_qrels,
            "dev_total_qrels": dev_total_qrels,
            "dev_positive_qrels": dev_positive_qrels,
            "train_qrel_rows_removed_by_doc_filter": train_filter_stats["qrel_rows_removed_by_doc_filter"],
            "dev_qrel_rows_removed_by_doc_filter": dev_filter_stats["qrel_rows_removed_by_doc_filter"],
        },
        "documents": {
            "unique_train_positive_docids_before_filter": len(train_docids),
            "unique_dev_positive_docids_before_filter": len(dev_docids),
            "unique_active_docids_before_filter": len(active_docids),
            "unique_active_docids_after_filter": len(valid_docids),
            **document_stats,
        },
    }

    with open(stats_output_path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Documents written: {stats['documents']['documents_written']:,}")
    print(f"Train queries:      {stats['queries']['train_queries_with_positive_qrels']:,}")
    print(f"Validation queries: {stats['queries']['dev_queries_with_positive_qrels']:,}")
    print(f"Train rows:         {stats['queries']['train_query_doc_rows']:,}")
    print(f"Validation rows:    {stats['queries']['dev_query_doc_rows']:,}")
    print(f"Empty-body docs skipped: {stats['documents']['empty_body_docs_skipped']:,}")
    print(f"Stats:              {stats_output_path}")


if __name__ == "__main__":
    main()
