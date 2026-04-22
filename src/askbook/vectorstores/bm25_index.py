"""Persistent BM25 index — pickle-backed, rank_bm25 powered."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Plus

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25PersistentIndex:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._chunk_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._tokens = data["tokens"]
        self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Plus(self._tokens) if self._tokens else None

    def add(self, items: list[tuple[str, str]]) -> None:
        existing = set(self._chunk_ids)
        for cid, text in items:
            if cid in existing:
                continue
            self._chunk_ids.append(cid)
            self._tokens.append(_tokenize(text))
            existing.add(cid)
        self._rebuild()

    def remove(self, chunk_ids: list[str]) -> None:
        drop = set(chunk_ids)
        keep_ids: list[str] = []
        keep_tokens: list[list[str]] = []
        for cid, toks in zip(self._chunk_ids, self._tokens, strict=True):
            if cid in drop:
                continue
            keep_ids.append(cid)
            keep_tokens.append(toks)
        self._chunk_ids = keep_ids
        self._tokens = keep_tokens
        self._rebuild()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        paired = sorted(
            zip(self._chunk_ids, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(s)) for cid, s in paired[:top_k] if s > 0.0]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as f:
            pickle.dump({"chunk_ids": self._chunk_ids, "tokens": self._tokens}, f)


__all__ = ["BM25PersistentIndex"]
