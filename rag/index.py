"""Persistent policy index: SQLite FTS5 for keywords plus FAISS for embeddings."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from rag.ingestion import Chunk

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
RetrievalMode = Literal["bm25", "vector", "hybrid"]
RRF_K = 60
CANDIDATES = 20
FUSION_WEIGHTS = {"bm25": 0.5, "vector": 1.0}


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
    similarity: float | None = None
    passage: str | None = None
    passage_similarity: float | None = None

    def metadata(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "section": self.section,
            "source": self.source,
            "snippet": self.snippet,
        }


def configured_mode() -> RetrievalMode | None:
    value = os.environ.get("HR_RETRIEVAL_MODE")
    if value in {"bm25", "vector", "hybrid"}:
        return value  # type: ignore[return-value]
    return None


class PolicyIndex:
    """Build and query a stable, portable policy index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.vector_path = path.with_suffix(".faiss")
        self.sentence_path = path.with_suffix(".sentences.npz")
        self._vectors: object | None = None
        self._vector_ids: list[str] | None = None
        self._sentences: tuple[object, list[str], dict[str, tuple[int, int]]] | None = None

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Policy index not found at {self.path}. Run `python -m rag build` first."
            )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def build(
        self,
        chunks: Iterable[Chunk],
        *,
        configuration: dict[str, object],
        embed: bool = True,
    ) -> int:
        """Replace the index atomically from chunks sorted by stable ID."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_vectors = self.vector_path.with_suffix(".faiss.tmp")
        temporary_sentences = self.path.with_suffix(".sentences.tmp.npz")
        for stale in (temporary, temporary_vectors, temporary_sentences):
            stale.unlink(missing_ok=True)
        ordered = sorted(chunks, key=lambda item: item.chunk_id)
        vector_metadata: dict[str, object] = {}
        if embed:
            vector_metadata = self._write_vectors(ordered, temporary_vectors, temporary_sentences)
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
                for chunk in ordered
            ]
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            metadata = {
                "chunk_count": len(rows),
                "configuration": {**configuration, **vector_metadata},
                "schema_version": 2,
            }
            connection.execute(
                "INSERT INTO index_metadata VALUES (?, ?)",
                ("manifest", json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
            connection.commit()
        finally:
            connection.close()
        temporary.replace(self.path)
        if embed:
            temporary_vectors.replace(self.vector_path)
            temporary_sentences.replace(self.sentence_path)
        else:
            self.vector_path.unlink(missing_ok=True)
            self.sentence_path.unlink(missing_ok=True)
        self._vectors = None
        self._vector_ids = None
        self._sentences = None
        return len(rows)

    @staticmethod
    def _write_vectors(
        chunks: list[Chunk], path: Path, sentence_path: Path
    ) -> dict[str, object]:
        import faiss
        import numpy as np

        from rag.embeddings import embed_passages, model_name

        vectors = embed_passages(
            [f"{chunk.title}. {chunk.section}. {chunk.text}" for chunk in chunks], bulk=True
        )
        index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, np.arange(len(chunks), dtype=np.int64))
        faiss.write_index(index, str(path))

        sentences: list[str] = []
        owners: list[str] = []
        for chunk in chunks:
            parts = [part.strip() for part in SENTENCE_RE.split(chunk.text) if part.strip()]
            sentences.extend(parts or [chunk.text])
            owners.extend([chunk.chunk_id] * max(len(parts), 1))
        np.savez(
            sentence_path,
            vectors=embed_passages(sentences, bulk=True),
            texts=np.array(sentences),
            chunk_ids=np.array(owners),
        )
        return {
            "embedding_model": model_name(),
            "embedding_dimensions": int(vectors.shape[1]),
            "vector_store": "faiss-flat-inner-product",
            "vector_chunk_ids": [chunk.chunk_id for chunk in chunks],
        }

    def manifest(self) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'manifest'"
            ).fetchone()
        if row is None:
            raise RuntimeError("Index manifest is missing")
        manifest = json.loads(row["value"])
        configuration = manifest.get("configuration", {})
        if isinstance(configuration, dict):
            configuration.pop("vector_chunk_ids", None)
        return manifest

    def has_vectors(self) -> bool:
        return self.vector_path.exists()

    def _load_vectors(self) -> tuple[object, list[str]]:
        if self._vectors is None or self._vector_ids is None:
            import faiss

            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM index_metadata WHERE key = 'manifest'"
                ).fetchone()
            self._vector_ids = list(json.loads(row["value"])["configuration"]["vector_chunk_ids"])
            self._vectors = faiss.read_index(str(self.vector_path))
        return self._vectors, self._vector_ids

    def _bm25_ids(self, query: str, limit: int, sources: Sequence[str] | None) -> list[str]:
        tokens = list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(query)))
        if not tokens:
            return []
        expression = " OR ".join(f'"{token.replace('"', '""')}"' for token in tokens[:40])
        source_filter = ""
        parameters: list[object] = [expression]
        if sources:
            source_filter = f"AND source IN ({', '.join('?' for _ in sources)})"
            parameters.extend(sources)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, bm25(chunks, 0.0, 0.0, 1.5, 1.2, 0.0, 1.0, 0.0) AS rank
                FROM chunks
                WHERE chunks MATCH ? {source_filter}
                ORDER BY rank ASC, chunk_id ASC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def _similarities(self, query_vector: object) -> dict[str, float]:
        index, chunk_ids = self._load_vectors()
        scores, positions = index.search(query_vector, len(chunk_ids))  # type: ignore[attr-defined]
        return {
            chunk_ids[position]: float(score)
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        }

    def _vector_ids_for(
        self, similarities: dict[str, float], limit: int, sources: Sequence[str] | None
    ) -> list[str]:
        allowed: set[str] | None = None
        if sources:
            with self._connect() as connection:
                allowed = {
                    row["chunk_id"]
                    for row in connection.execute(
                        f"SELECT chunk_id FROM chunks WHERE source IN ({', '.join('?' for _ in sources)})",
                        list(sources),
                    )
                }
        ranked = sorted(similarities, key=lambda chunk_id: (-similarities[chunk_id], chunk_id))
        ordered = [chunk_id for chunk_id in ranked if allowed is None or chunk_id in allowed]
        return ordered[:limit]

    def _load_sentences(self) -> tuple[object, list[str], dict[str, tuple[int, int]]]:
        if self._sentences is None:
            import numpy as np

            with np.load(self.sentence_path) as stored:
                vectors = stored["vectors"]
                texts = [str(text) for text in stored["texts"]]
                owners = [str(owner) for owner in stored["chunk_ids"]]
            spans: dict[str, tuple[int, int]] = {}
            for position, owner in enumerate(owners):
                start, _ = spans.get(owner, (position, position))
                spans[owner] = (start, position + 1)
            self._sentences = (vectors, texts, spans)
        return self._sentences

    def _passages(
        self, query_vector: object, chunk_ids: Sequence[str]
    ) -> list[tuple[str, float] | None]:
        """Pick the sentence in each chunk closest in meaning to the query."""

        import numpy as np

        if not self.sentence_path.exists():
            return [None] * len(chunk_ids)
        vectors, texts, spans = self._load_sentences()
        query = np.asarray(query_vector)[0]
        passages: list[tuple[str, float] | None] = []
        for chunk_id in chunk_ids:
            if chunk_id not in spans:
                passages.append(None)
                continue
            start, end = spans[chunk_id]
            scores = vectors[start:end] @ query  # type: ignore[index]
            best = int(np.argmax(scores))
            passages.append((texts[start + best], round(float(scores[best]), 4)))
        return passages

    def _rows(self, chunk_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, document_id, title, section, source, text, snippet
                FROM chunks WHERE chunk_id IN ({', '.join('?' for _ in chunk_ids)})
                """,
                list(chunk_ids),
            ).fetchall()
        return {row["chunk_id"]: row for row in rows}

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: Sequence[str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[SearchResult]:
        """Return ranked chunks, optionally limited to some source files.

        `hybrid` fuses BM25 and embedding rankings with weighted reciprocal
        rank fusion. Without a vector index, every mode falls back to BM25.
        `bm25` never loads the embedding model, so results carry no
        similarity or passage.
        """

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        mode = mode or configured_mode() or "hybrid"
        if not self.has_vectors():
            mode = "bm25"
        query_vector = None
        similarities: dict[str, float] = {}
        if mode != "bm25":
            from rag.embeddings import embed_query

            query_vector = embed_query(query)
            similarities = self._similarities(query_vector)
        rankings: list[tuple[float, list[str]]] = []
        if mode in {"bm25", "hybrid"}:
            weight = FUSION_WEIGHTS["bm25"] if mode == "hybrid" else 1.0
            rankings.append((weight, self._bm25_ids(query, CANDIDATES, sources)))
        if mode in {"vector", "hybrid"}:
            rankings.append(
                (FUSION_WEIGHTS["vector"], self._vector_ids_for(similarities, CANDIDATES, sources))
            )
        fused: dict[str, float] = {}
        for weight, ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        rows = self._rows([chunk_id for chunk_id, _ in ordered])
        passages: list[tuple[str, float] | None] = [None] * len(ordered)
        if query_vector is not None and ordered:
            passages = self._passages(query_vector, [chunk_id for chunk_id, _ in ordered])
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=rows[chunk_id]["document_id"],
                title=rows[chunk_id]["title"],
                section=rows[chunk_id]["section"],
                source=rows[chunk_id]["source"],
                text=rows[chunk_id]["text"],
                snippet=rows[chunk_id]["snippet"],
                score=round(score, 8),
                similarity=round(similarities[chunk_id], 4) if chunk_id in similarities else None,
                passage=passage[0] if passage else None,
                passage_similarity=passage[1] if passage else None,
            )
            for (chunk_id, score), passage in zip(ordered, passages)
        ]

    def export_results(
        self,
        query: str,
        *,
        top_k: int = 5,
        sources: Sequence[str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[dict[str, object]]:
        return [
            asdict(result)
            for result in self.search(query, top_k=top_k, sources=sources, mode=mode)
        ]
