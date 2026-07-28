from __future__ import annotations

from ingest import SourceUnit, chunk_source_units, count_tokens, decode_tokens, encode_text


def unit(text: str, source: str = "notes.txt") -> SourceUnit:
    return SourceUnit(
        text=text,
        source=source,
        location="Lines 1-10",
        category="Reading Materials",
        unit_index=0,
    )


def test_short_text_produces_one_chunk():
    chunks = chunk_source_units([unit("short text")], 50, 10)
    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens(chunks[0].text)


def test_exact_size_produces_one_chunk():
    tokens = encode_text("one two three four five six")
    text = decode_tokens(tokens)
    chunks = chunk_source_units([unit(text)], len(tokens), 1)
    assert len(chunks) == 1


def test_multiple_chunks_have_expected_token_overlap():
    text = " ".join(f"term{index}" for index in range(80))
    original = encode_text(text)
    size, overlap = 20, 5
    chunks = chunk_source_units([unit(text)], size, overlap)
    expected = []
    start = 0
    while start < len(original):
        end = min(start + size, len(original))
        expected.append(" ".join(decode_tokens(original[start:end]).split()))
        if end == len(original):
            break
        start = end - overlap
    assert [" ".join(chunk.text.split()) for chunk in chunks] == expected
    assert all(chunk.token_count <= size for chunk in chunks)


def test_zero_overlap_cannot_loop_forever():
    chunks = chunk_source_units(
        [unit(" ".join(f"word{index}" for index in range(200)))],
        10,
        0,
    )
    assert 1 < len(chunks) < 100
    assert all(chunk.text.strip() for chunk in chunks)


def test_ids_and_output_are_deterministic():
    units = [unit("deterministic chunk text " * 20)]
    first = chunk_source_units(units, 20, 5)
    second = chunk_source_units(units, 20, 5)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_metadata_is_preserved_without_cross_document_chunks():
    units = [
        unit("alpha " * 40, "alpha.txt"),
        unit("beta " * 40, "beta.txt"),
    ]
    chunks = chunk_source_units(units, 15, 3)
    assert {chunk.source for chunk in chunks} == {"alpha.txt", "beta.txt"}
    for chunk in chunks:
        if chunk.source == "alpha.txt":
            assert "beta" not in chunk.text
        else:
            assert "alpha" not in chunk.text


def test_long_single_page_preserves_location():
    source = SourceUnit(
        text="evidence " * 500,
        source="paper.pdf",
        location="Page 7",
        category="Reading Materials",
        unit_index=6,
    )
    chunks = chunk_source_units([source], 50, 10)
    assert len(chunks) > 1
    assert {chunk.location for chunk in chunks} == {"Page 7"}
