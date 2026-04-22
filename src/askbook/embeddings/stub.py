"""Deterministic stub embedder for unit tests — no model weights involved."""

from __future__ import annotations

import hashlib
import struct


class StubEmbedder:
    model_name: str = "stub"

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_vec(self, text: str, salt: str) -> list[float]:
        h = hashlib.sha256((salt + "\x1f" + text).encode()).digest()
        needed = self._dimension * 4
        raw = (h * ((needed // len(h)) + 1))[:needed]
        return [struct.unpack("<f", raw[i : i + 4])[0] for i in range(0, needed, 4)]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vec(text, "query")

    def embed_passage(self, text: str) -> list[float]:
        return self._hash_vec(text, "passage")

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        fn = self.embed_query if is_query else self.embed_passage
        return [fn(t) for t in texts]


__all__ = ["StubEmbedder"]
