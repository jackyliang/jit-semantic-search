"""CLI tool for JIT Semantic Search.

Usage:
    # Search from stdin (pipe JSON array of documents)
    echo '["doc1", "doc2", "doc3"]' | python -m jit_search.cli "search query" --strategy cascade --top-k 5

    # Search from a JSON file
    python -m jit_search.cli "search query" --file documents.json --strategy cascade --top-k 5

    # Search structured objects from JSON file
    python -m jit_search.cli "search query" --file tickets.json --text-fields subject,description --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jit-search",
        description="JIT Semantic Search — search documents from the command line",
    )
    parser.add_argument("query", help="The search query")
    parser.add_argument(
        "--file", "-f",
        help="Path to a JSON file containing documents (array of strings) or objects (array of dicts). "
             "If omitted, reads JSON from stdin.",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="cascade",
        help="Search strategy to use (default: cascade)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--text-fields",
        help="Comma-separated field names to extract text from when searching objects "
             "(e.g. --text-fields subject,description)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Load data ----
    if args.file:
        with open(args.file) as fh:
            data = json.load(fh)
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            parser.error("No input provided. Pipe JSON via stdin or use --file.")
        data = json.loads(raw)

    if not isinstance(data, list) or len(data) == 0:
        parser.error("Input must be a non-empty JSON array.")

    # ---- Determine mode: plain strings vs structured objects ----
    text_fields = [f.strip() for f in args.text_fields.split(",")] if args.text_fields else None

    if text_fields:
        # Structured object search
        objects = data
        documents = []
        for obj in objects:
            parts = [str(obj.get(f, "")) for f in text_fields]
            documents.append(" ".join(parts))
    else:
        # Plain document search
        if not all(isinstance(d, str) for d in data):
            parser.error(
                "Input contains non-string items. Use --text-fields to specify which "
                "fields to extract text from when searching objects."
            )
        objects = None
        documents = data

    # ---- Run search ----
    from jit_search import JITSearch

    searcher = JITSearch(strategy=args.strategy)
    results, elapsed_ms = searcher.search_timed(args.query, documents, args.top_k)

    # ---- Format output ----
    output = {
        "query": args.query,
        "strategy": args.strategy,
        "latency_ms": round(elapsed_ms, 1),
        "num_documents": len(documents),
        "results": [],
    }

    for r in results:
        item: dict = {"index": r.index, "score": round(r.score, 4), "document": r.document}
        if objects is not None:
            item["object"] = objects[r.index]
        output["results"].append(item)

    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
