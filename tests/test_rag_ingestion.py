from pathlib import Path

import pytest

from rag.ingestion import chunk_documents, load_documents


POLICY_DIR = Path(__file__).parents[1] / "policies"


def test_loads_both_supported_policy_formats_with_stable_ids() -> None:
    first = load_documents(POLICY_DIR)
    second = load_documents(POLICY_DIR)

    assert len(first) == 12
    assert {Path(document.source).suffix for document in first} == {".md", ".txt"}
    assert [(document.document_id, document.source) for document in first] == [
        (document.document_id, document.source) for document in second
    ]
    assert all(document.document_id.startswith("policy-") for document in first)


def test_heading_aware_chunks_are_deterministic_and_keep_required_metadata() -> None:
    documents = load_documents(POLICY_DIR)
    first = chunk_documents(documents, chunk_size=90, overlap=15)
    second = chunk_documents(documents, chunk_size=90, overlap=15)

    assert first == second
    assert len(first) > len(documents)
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(
        (
            chunk.document_id
            and chunk.title
            and chunk.section
            and chunk.source
            and chunk.snippet
        )
        for chunk in first
    )
    assert any(chunk.section == "Eligibility" for chunk in first)
    assert any(chunk.section == "Requesting An Accommodation" for chunk in first)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(19, 0), (100, -1), (100, 100)],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=chunk_size, overlap=overlap)
