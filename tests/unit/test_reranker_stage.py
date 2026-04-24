"""Tests for CrossEncoderRerankNode and LLMFineRerankNode."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import RetrievalResult
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.rerankers.stub import StubReranker


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _results(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=str(i),
            score=float(i),
            snippet="s",
            metadata={},
            retrieval_method="rrf",
        )
        for i in range(n)
    ]


def test_cross_encoder_reduces_to_top_k() -> None:
    node = CrossEncoderRerankNode(
        reranker=StubReranker(), top_k=3, trace_writer=_make_trace()
    )
    ctx = {"query": "q", "retrieval_results": _results(10)}
    out = node(ctx)  # type: ignore[arg-type]
    assert len(out["retrieval_results"]) == 3
    assert all(r.retrieval_method == "reranked" for r in out["retrieval_results"])


def test_cross_encoder_empty_results_passthrough() -> None:
    node = CrossEncoderRerankNode(
        reranker=StubReranker(), top_k=3, trace_writer=_make_trace()
    )
    ctx = {"query": "q", "retrieval_results": []}
    out = node(ctx)  # type: ignore[arg-type]
    assert out["retrieval_results"] == []


def test_llm_fine_rerank_disabled_passthrough() -> None:
    node = LLMFineRerankNode(enabled=False, trace_writer=_make_trace())
    ctx = {"query": "q", "retrieval_results": _results(5)}
    out = node(ctx)  # type: ignore[arg-type]
    assert out is ctx  # same object (no copy when passthrough)


def test_reranker_nodes_are_idempotent() -> None:
    node = CrossEncoderRerankNode(
        reranker=StubReranker(), top_k=5, trace_writer=_make_trace()
    )
    ctx = {"query": "q", "retrieval_results": _results(8)}
    out1 = node(ctx)  # type: ignore[arg-type]
    out2 = node(ctx)  # type: ignore[arg-type]
    assert [r.chunk_id for r in out1["retrieval_results"]] == [
        r.chunk_id for r in out2["retrieval_results"]
    ]
