"""Batched OpenAI embeddings and a compact normalized NumPy index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ingest import DocumentChunk


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced safely."""


class InvalidIndexError(ValueError):
    """Raised when vector-index invariants are violated."""


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix without mutating its input."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise InvalidIndexError("Vectors must be a 1-D vector or 2-D matrix.")
    norms = np.linalg.norm(array, axis=-1, keepdims=array.ndim == 2)
    if np.any(norms == 0):
        raise InvalidIndexError("Zero-length vectors cannot be normalized.")
    return array / norms


def embed_texts(
    texts: list[str],
    client,
    model: str,
    batch_size: int,
) -> np.ndarray:
    """Embed non-empty texts in batches, preserve order, and normalize."""

    if not texts:
        raise EmbeddingError("At least one text is required for embedding.")
    if batch_size <= 0:
        raise EmbeddingError("batch_size must be positive.")
    if any(not text.strip() for text in texts):
        raise EmbeddingError("Empty texts cannot be embedded.")

    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        data = sorted(
            response.data,
            key=lambda item: getattr(item, "index", 0),
        )
        if len(data) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings but received {len(data)}."
            )
        rows.extend(item.embedding for item in data)
    if len(rows) != len(texts):
        raise EmbeddingError("Embedding response order or count was invalid.")
    try:
        return normalize_vectors(np.asarray(rows, dtype=np.float32))
    except (ValueError, InvalidIndexError) as exc:
        raise EmbeddingError("Embedding vectors were malformed.") from exc


class NumpyVectorIndex:
    """In-memory cosine-similarity search over normalized embeddings."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise InvalidIndexError("Embedding matrix must be 2-D.")
        if len(chunks) != matrix.shape[0]:
            raise InvalidIndexError(
                "The number of chunks must equal the embedding row count."
            )
        self.chunks = list(chunks)
        self.embeddings = (
            normalize_vectors(matrix) if matrix.shape[0] else matrix.copy()
        )

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> list[RetrievalResult]:
        if k <= 0:
            raise InvalidIndexError("k must be positive.")
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise InvalidIndexError("Query vector must be one-dimensional.")
        if query.shape[0] != self.embeddings.shape[1]:
            raise InvalidIndexError(
                "Query dimension does not match the embedding index."
            )
        query = normalize_vectors(query)
        scores = self.embeddings @ query
        limit = min(k, len(self.chunks))
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            RetrievalResult(
                chunk=self.chunks[int(index)],
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        ]

    def search(
        self,
        query: str,
        embed_query_fn: Callable[[str], np.ndarray],
        k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        return self.search_by_vector(embed_query_fn(query), k)

