from __future__ import annotations

from ingest import DocumentChunk
from llm import (
    REFUSAL_TEXT,
    authoritative_source_mapping,
    extract_citation_numbers,
    validate_citations,
)
from retrieve import RetrievalResult


def result(number: int) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=str(number),
        text="Evidence",
        source=f"source-{number}.txt",
        location=f"Lines {number}-{number + 1}",
        category="Reading Materials",
        unit_index=0,
        chunk_index=0,
        token_count=1,
    )
    return RetrievalResult(chunk, 0.9, number)


def test_extracts_valid_multiple_and_duplicate_citations():
    assert extract_citation_numbers("Claim [1], more [2], again [1].") == {1, 2}


def test_valid_answer_requires_sources_section():
    validation = validate_citations("Claim [1].\n\nSources:\n[1]", [result(1)])
    assert validation.is_valid


def test_zero_and_out_of_range_are_invalid():
    validation = validate_citations(
        "Claims [0] [3].\n\nSources:\n[0] [3]",
        [result(1), result(2)],
    )
    assert validation.invalid_numbers == {0, 3}
    assert not validation.is_valid


def test_no_citation_is_invalid_for_non_refusal():
    assert not validate_citations("An uncited answer.", [result(1)]).is_valid


def test_refusal_does_not_require_citation():
    assert validate_citations(REFUSAL_TEXT, [result(1)]).is_valid


def test_authoritative_mapping_uses_retrieval_metadata():
    mapping = authoritative_source_mapping([result(1), result(2)], {2})
    assert mapping == [(2, "source-2.txt", "Lines 2-3")]

