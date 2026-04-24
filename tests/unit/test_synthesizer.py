"""Tests for AnswerSynthesizerNode."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import RetrievalResult
from askbook.providers.stub import StubLLMProvider
from askbook.query.synthesizer import AnswerSynthesizerNode


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _rr(cid: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid, score=0.9, snippet="snip", metadata={}, retrieval_method="rrf"
    )


def test_synthesizer_produces_answer_with_citations() -> None:
    node = AnswerSynthesizerNode(
        llm=StubLLMProvider(canned_response="Answer text"), trace_writer=_make_trace()
    )
    ctx = {
        "query": "Q",
        "retrieval_results": [_rr("c1"), _rr("c2")],
        "pipeline_trace_id": "trace-1",
    }
    out = node(ctx)  # type: ignore[arg-type]
    assert out["answer"].text == "Answer text"
    assert [c.chunk_id for c in out["answer"].citations] == ["c1", "c2"]
    assert out["answer"].pipeline_trace_id == "trace-1"


def test_synthesizer_empty_results_returns_fallback() -> None:
    node = AnswerSynthesizerNode(llm=StubLLMProvider(), trace_writer=_make_trace())
    ctx = {"query": "Q", "retrieval_results": [], "pipeline_trace_id": "t2"}
    out = node(ctx)  # type: ignore[arg-type]
    assert "资料不足" in out["answer"].text
    assert out["answer"].citations == []


def test_synthesizer_uses_rewritten_query_when_present() -> None:
    node = AnswerSynthesizerNode(
        llm=StubLLMProvider(canned_response="ok"), trace_writer=_make_trace()
    )
    ctx = {
        "query": "original",
        "rewritten_query": "rewritten",
        "retrieval_results": [_rr("x")],
        "pipeline_trace_id": "t3",
    }
    out = node(ctx)  # type: ignore[arg-type]
    assert out["answer"].text == "ok"


def test_synthesizer_trace_id_threading() -> None:
    node = AnswerSynthesizerNode(
        llm=StubLLMProvider(canned_response="hi"), trace_writer=_make_trace()
    )
    ctx = {"query": "q", "retrieval_results": [_rr("a")], "pipeline_trace_id": "myid"}
    out = node(ctx)  # type: ignore[arg-type]
    assert out["answer"].pipeline_trace_id == "myid"
