"""Integration tests for query rewrite functionality."""

from __future__ import annotations

from askbook.observability.null_trace import NullTraceWriter
from askbook.providers.stub import StubLLMProvider
from askbook.query.rewriter import QueryRewriterNode


def test_rewrite_enabled_modifies_query() -> None:
    """enable_rewrite=True with LLM -> query is rewritten, original_query preserved."""
    stub_llm = StubLLMProvider(model="test")
    node = QueryRewriterNode(llm=stub_llm, enabled=True, trace_writer=NullTraceWriter())
    result = node.run({"query": "简单提问"})
    assert result["original_query"] == "简单提问"
    assert result["rewritten_query"] != "简单提问"


def test_rewrite_disabled_passthrough() -> None:
    """enable_rewrite=False -> query unchanged, original_query == rewritten_query."""
    stub_llm = StubLLMProvider(model="test")
    node = QueryRewriterNode(
        llm=stub_llm, enabled=False, trace_writer=NullTraceWriter()
    )
    result = node.run({"query": "hello"})
    assert result["rewritten_query"] == "hello"
    assert result["original_query"] == "hello"


def test_rewrite_no_llm_passthrough() -> None:
    """enable_rewrite=True but llm=None -> query unchanged."""
    node = QueryRewriterNode(llm=None, enabled=True, trace_writer=NullTraceWriter())
    result = node.run({"query": "test query"})
    assert result["rewritten_query"] == "test query"
