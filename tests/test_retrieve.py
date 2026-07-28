from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ingest import DocumentChunk
from retrieve import (
    EmbeddingError,
    InvalidIndexError,
    NumpyVectorIndex,
    embed_texts,
    normalize_vectors,
)


def chunk(name: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=name,
        text=name,
        source=f"{name}.txt",
        location="Line 1",
        category="Reading Materials",
        unit_index=0,
        chunk_index=0,
        token_count=1,
    )


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
                for index, _ in enumerate(input)
            ]
        )


def test_embedding_batches_and_normalizes():
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    matrix = embed_texts(["a", "b", "c"], client, "embed-model", 2)
    assert len(embeddings.calls) == 2
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(3))


def test_empty_embedding_input_is_rejected():
    with pytest.raises(EmbeddingError):
        embed_texts([], SimpleNamespace(), "model", 5)


def test_normalization_rejects_zero_norm():
    with pytest.raises(InvalidIndexError):
        normalize_vectors(np.array([0.0, 0.0]))


def test_ranking_and_k_clamping():
    chunks = [chunk("x"), chunk("y"), chunk("z")]
    matrix = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    index = NumpyVectorIndex(chunks, matrix)
    results = index.search_by_vector(np.array([1, 0]), 10)
    assert [item.chunk.chunk_id for item in results] == ["x", "y", "z"]
    assert [item.rank for item in results] == [1, 2, 3]


def test_stable_tie_order():
    chunks = [chunk("first"), chunk("second")]
    index = NumpyVectorIndex(chunks, np.array([[1, 0], [1, 0]]))
    results = index.search_by_vector(np.array([1, 0]), 2)
    assert [item.chunk.chunk_id for item in results] == ["first", "second"]


def test_empty_index_returns_no_results():
    index = NumpyVectorIndex([], np.empty((0, 2), dtype=np.float32))
    assert index.search_by_vector(np.array([1, 0]), 4) == []


def test_index_dimension_mismatch():
    index = NumpyVectorIndex([chunk("x")], np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(InvalidIndexError, match="dimension"):
        index.search_by_vector(np.array([1, 0, 0]), 1)


def test_row_count_mismatch():
    with pytest.raises(InvalidIndexError, match="row count"):
        NumpyVectorIndex([chunk("x")], np.empty((2, 2)))

