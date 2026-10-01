"""Local sentence embeddings for semantic policy retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = Path(os.environ.get("HR_EMBEDDING_CACHE", PROJECT_ROOT / "data" / "models"))


def model_name() -> str:
    return os.environ.get("HR_EMBEDDING_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=4)
def _embedder(name: str, threads: int | None) -> "TextEmbedding":
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=str(MODEL_CACHE), threads=threads)


def _normalized(vectors: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed_passages(
    texts: Sequence[str], *, name: str | None = None, bulk: bool = False
) -> np.ndarray:
    """Return unit-length passage vectors, one row per text.

    `bulk` uses every core for index builds; request-time calls share the
    single-thread session that queries use, so only one model is loaded.
    """

    embedder = _embedder(name or model_name(), None if bulk else 1)
    return _normalized(list(embedder.embed(list(texts), batch_size=32)))


def embed_query(text: str, *, name: str | None = None) -> np.ndarray:
    """Return a unit-length query vector shaped (1, dimensions).

    Queries use one thread to keep the long-lived MCP server's memory small.
    """

    return _normalized(list(_embedder(name or model_name(), 1).query_embed([text])))
