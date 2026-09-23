"""Citation assembly and a deterministic extractive answer path."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.guardrails import has_sufficient_evidence, validate_query
from rag.index import PolicyIndex, SearchResult

REFUSAL = (
    "I could not find enough support in the synthetic HR policy corpus to answer that. "
    "Please contact an HR representative for authoritative guidance."
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    citation_id: str
    document_id: str
    title: str
    section: str
    source: str
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    grounded: bool
    label: str


def _best_sentence(result: SearchResult, query: str) -> str:
    terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 4}
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(result.text) if sentence.strip()]
    if not sentences:
        return result.snippet
    return max(
        sentences,
        key=lambda sentence: (
            len(terms & set(re.findall(r"[a-z0-9]+", sentence.lower()))),
            -len(sentence),
        ),
    )


def citations_for(results: list[SearchResult]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen_documents: set[str] = set()
    for result in results:
        if result.document_id in seen_documents:
            continue
        seen_documents.add(result.document_id)
        citations.append(
            Citation(
                citation_id=f"P{len(citations) + 1}",
                document_id=result.document_id,
                title=result.title,
                section=result.section,
                source=result.source,
                snippet=result.snippet,
            )
        )
    return tuple(citations)


def answer_query(index: PolicyIndex, query: str, *, top_k: int = 5) -> Answer:
    """Return a concise policy-labeled answer with validated inline citations."""

    normalized = validate_query(query)
    results = index.search(normalized, top_k=top_k)
    if not results or not has_sufficient_evidence(normalized, [result.text for result in results]):
        return Answer(text=REFUSAL, citations=(), grounded=False, label="Escalation")

    citations = citations_for(results)
    citation_by_document = {citation.document_id: citation for citation in citations}
    claims: list[str] = []
    used: set[str] = set()
    for result in results:
        citation = citation_by_document[result.document_id]
        if citation.citation_id in used:
            continue
        used.add(citation.citation_id)
        sentence = _best_sentence(result, normalized)
        claims.append(f"{sentence} [{citation.citation_id}]")
        if len(claims) == 3:
            break
    text = "Policy guidance: " + " ".join(claims)
    used_citations = tuple(citation for citation in citations if citation.citation_id in used)
    return Answer(
        text=text,
        citations=used_citations,
        grounded=True,
        label="Policy guidance",
    )
