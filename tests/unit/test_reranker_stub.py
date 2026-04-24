"""Tests for StubReranker."""

from __future__ import annotations

import asyncio

from askbook.core.models import RetrievalResult
from askbook.rerankers.stub import StubReranker


def _r(cid: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid, score=score, snippet="s", metadata={}, retrieval_method="bm25"
    )


def test_stub_reranker_sorts_score_desc() -> None:
    rr = StubReranker()
    results = [_r("A", 0.3), _r("B", 0.8), _r("C", 0.5)]
    out = rr.rerank("q", results, top_k=2)
    assert [r.chunk_id for r in out] == ["B", "C"]
    assert all(r.retrieval_method == "reranked" for r in out)


def test_stub_reranker_respects_top_k() -> None:
    rr = StubReranker()
    results = [_r(str(i), float(i)) for i in range(10)]
    assert len(rr.rerank("q", results, top_k=3)) == 3


def test_stub_reranker_empty_returns_empty() -> None:
    assert StubReranker().rerank("q", [], top_k=5) == []


def test_stub_reranker_arerank_matches_sync() -> None:
    rr = StubReranker()
    results = [_r("X", 0.9), _r("Y", 0.1)]
    sync_out = rr.rerank("q", results, top_k=2)
    async_out = asyncio.run(rr.arerank("q", results, top_k=2))
    assert [r.chunk_id for r in sync_out] == [r.chunk_id for r in async_out]
