"""Deterministic bag-of-words embedder used in place of fastembed in tests."""

import hashlib

import numpy as np

DIM = 64


class StubEmbedder:
    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def embed(self, texts):
        return (self._vec(t) for t in texts)

    query_embed = embed

    def __call__(self):
        return self
