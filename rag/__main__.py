"""Reproducible command line interface for the policy index."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.answering import answer_query
from rag.index import PolicyIndex
from rag.ingestion import chunk_documents, load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES = PROJECT_ROOT / "policies"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "policy_index.sqlite3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query the synthetic HR policy index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the persistent index.")
    build.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--chunk-size", type=int, default=180)
    build.add_argument("--overlap", type=int, default=30)

    search = subparsers.add_parser("search", help="Search indexed policy chunks.")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--top-k", type=int, default=5)

    answer = subparsers.add_parser("answer", help="Create a cited extractive policy answer.")
    answer.add_argument("query")
    answer.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    answer.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    index = PolicyIndex(args.index)
    if args.command == "build":
        documents = load_documents(args.policies)
        chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
        count = index.build(
            chunks,
            configuration={
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "policy_directory": args.policies.resolve().as_posix(),
                "retrieval": "sqlite-fts5-bm25",
            },
        )
        print(json.dumps({"documents": len(documents), "chunks": count, "index": str(args.index)}))
    elif args.command == "search":
        print(json.dumps(index.export_results(args.query, top_k=args.top_k), indent=2))
    else:
        print(json.dumps(asdict(answer_query(index, args.query, top_k=args.top_k)), indent=2))


if __name__ == "__main__":
    main()
