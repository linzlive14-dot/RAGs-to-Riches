"""Persistent deterministic policy index using Python's bundled SQLite FTS5."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rag.ingestion import Chunk

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    text: str
    snippet: str
    score: float

    def metadata(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "snippet": self.snippet,
        }


class PolicyIndex:
    """Build and query a stable, portable full-text policy index."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Policy index not found at {self.path}. Run `python -m rag build` first."
            )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def build(self, chunks: Iterable[Chunk], *, configuration: dict[str, object]) -> int:
        """Replace the index atomically from chunks sorted by stable ID."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        connection = sqlite3.connect(temporary)
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    section,
                    source UNINDEXED,
                    text,
                    snippet UNINDEXED,
                    tokenize = "unicode61 remove_diacritics 2"
                )
                """
            )
            connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            rows = [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.title,
                    chunk.section,
                    chunk.source,
                    chunk.text,
                    chunk.snippet,
                )
                for chunk in sorted(chunks, key=lambda item: item.chunk_id)
            ]
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            metadata = {
                "chunk_count": len(rows),
                "configuration": configuration,
                "schema_version": 1,
            }
            connection.execute(
                "INSERT INTO index_metadata VALUES (?, ?)",
                ("manifest", json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
            connection.commit()
        finally:
            connection.close()
        temporary.replace(self.path)
        return len(rows)

    def manifest(self) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'manifest'"
            ).fetchone()
        if row is None:
            raise RuntimeError("Index manifest is missing")
        return json.loads(row["value"])

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return BM25-ranked chunks with deterministic tie-breaking."""

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        tokens = list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(query)))
        if not tokens:
            return []
        expression = " OR ".join(f'"{token.replace('"', '""')}"' for token in tokens[:40])
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, title, section, source, text, snippet,
                       bm25(chunks, 0.0, 0.0, 1.5, 1.2, 0.0, 1.0, 0.0) AS rank
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank ASC, chunk_id ASC
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                title=row["title"],
                section=row["section"],
                source=row["source"],
                text=row["text"],
                snippet=row["snippet"],
                score=round(-float(row["rank"]), 8),
            )
            for row in rows
        ]

    def export_results(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        return [asdict(result) for result in self.search(query, top_k=top_k)]
