"""Deterministic stand-ins for the fastembed models, used in tests."""

import hashlib
from types import SimpleNamespace

import numpy as np

DIM = 64


def _bucket(word: str, size: int) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % size


class StubEmbedder:
    """Bag-of-words dense embedder."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            v[_bucket(word, DIM)] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def embed(self, texts):
        return (self._vec(t) for t in texts)

    query_embed = embed

    def __call__(self):
        return self


class StubSparseEmbedder:
    """Term counts over hashed word ids, shaped like fastembed's SparseEmbedding."""

    def _vec(self, text: str):
        counts: dict[int, float] = {}
        for word in text.lower().split():
            i = _bucket(word, 100_000)
            counts[i] = counts.get(i, 0.0) + 1.0
        ids = sorted(counts)
        return SimpleNamespace(
            indices=np.array(ids, dtype=np.int64),
            values=np.array([counts[i] for i in ids], dtype=np.float32),
        )

    def embed(self, texts):
        return (self._vec(t) for t in texts)

    query_embed = embed

    def __call__(self):
        return self


class StubReranker:
    """Scores a document by how many query words it contains."""

    def rerank(self, query: str, documents):
        words = set(query.lower().split())
        return (float(len(words & set(d.lower().split()))) for d in documents)

    def __call__(self):
        return self
