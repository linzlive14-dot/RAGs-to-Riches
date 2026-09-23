"""Deterministic, heading-aware ingestion for Markdown and text policies."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".md", ".txt"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TXT_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 /&(),'-]{2,80}$")
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    text: str
    snippet: str
    ordinal: int


def _stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def _title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
        stripped = line.strip()
        if stripped:
            return stripped.title() if path.suffix.lower() == ".txt" else stripped
    return path.stem.replace("_", " ").replace("-", " ").title()


def load_documents(policy_dir: Path) -> list[SourceDocument]:
    """Load policy files in stable path order with stable identifiers."""

    root = policy_dir.resolve()
    documents: list[SourceDocument] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _clean_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        source = path.relative_to(root).as_posix()
        documents.append(
            SourceDocument(
                document_id=f"policy-{_stable_id(source)}",
                title=_title_from_text(path, text),
                source=source,
                text=text,
            )
        )
    return documents


def _sections(document: SourceDocument) -> Iterable[tuple[str, str]]:
    current_heading = document.title
    current_lines: list[str] = []

    def flush() -> tuple[str, str] | None:
        body = "\n".join(current_lines).strip()
        return (current_heading, body) if body else None

    for line in document.text.splitlines():
        markdown = HEADING_RE.match(line)
        plain = (
            document.source.endswith(".txt")
            and TXT_HEADING_RE.match(line.strip())
            and len(line.split()) <= 10
        )
        if markdown or plain:
            section = flush()
            if section:
                yield section
            current_heading = markdown.group(2).strip() if markdown else line.strip().title()
            current_lines = []
        else:
            current_lines.append(line)
    section = flush()
    if section:
        yield section


def _word_windows(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = WORD_RE.findall(text)
    if not words:
        return
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            yield " ".join(window)
        if start + chunk_size >= len(words):
            break


def chunk_documents(
    documents: Iterable[SourceDocument],
    *,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Split documents by heading and then fixed word windows."""

    if chunk_size < 20:
        raise ValueError("chunk_size must be at least 20 words")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda item: item.source):
        ordinal = 0
        for section, body in _sections(document):
            for text in _word_windows(body, chunk_size, overlap):
                identity = f"{document.document_id}\0{section}\0{ordinal}\0{text}"
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk-{_stable_id(identity, 24)}",
                        document_id=document.document_id,
                        title=document.title,
                        section=section,
                        source=document.source,
                        text=text,
                        snippet=text[:240].rstrip(),
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
    return chunks
