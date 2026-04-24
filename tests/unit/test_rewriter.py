"""Tests for QueryRewriterNode and HyDENode."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.providers.stub import StubLLMProvider
from askbook.query.hyde import HyDENode
from askbook.query.rewriter import QueryRewriterNode


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def test_query_rewriter_passthrough_fills_rewritten_query() -> None:
    node = QueryRewriterNode(enabled=False, trace_writer=_make_trace())
    out = node({"query": "foo", "collection": "col"})  # type: ignore[arg-type]
    assert out["rewritten_query"] == "foo"


def test_query_rewriter_disabled_even_with_llm_uses_original() -> None:
    node = QueryRewriterNode(
        enabled=False,
        llm=StubLLMProvider(canned_response="refined"),
        trace_writer=_make_trace(),
    )
    out = node({"query": "original", "collection": "col"})  # type: ignore[arg-type]
    assert out["rewritten_query"] == "original"


def test_query_rewriter_enabled_calls_llm() -> None:
    node = QueryRewriterNode(
        enabled=True,
        llm=StubLLMProvider(canned_response="refined foo"),
        trace_writer=_make_trace(),
    )
    out = node({"query": "foo", "collection": "col"})  # type: ignore[arg-type]
    assert out["rewritten_query"] == "refined foo"


def test_query_rewriter_enabled_no_llm_falls_back_to_original() -> None:
    node = QueryRewriterNode(enabled=True, llm=None, trace_writer=_make_trace())
    out = node({"query": "bar", "collection": "col"})  # type: ignore[arg-type]
    assert out["rewritten_query"] == "bar"


def test_hyde_disabled_leaves_context_unchanged() -> None:
    node = HyDENode(enabled=False, trace_writer=_make_trace())
    ctx = {"query": "q", "collection": "col"}
    out = node(ctx)  # type: ignore[arg-type]
    assert out["query"] == "q"
    assert "hyde_query" not in out
