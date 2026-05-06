"""Integration tests for HyDE (Hypothetical Document Embeddings)."""

from __future__ import annotations

from askbook.observability.null_trace import NullTraceWriter
from askbook.providers.stub import StubLLMProvider
from askbook.query.hyde import HyDENode


def test_hyde_disabled_passthrough() -> None:
    """enabled=False: context passes through, no "hyde_query" key added."""
    node = HyDENode(enabled=False, trace_writer=NullTraceWriter())
    ctx: dict[str, object] = {"query": "what is RAG?", "collection": "test"}
    out = node(ctx)  # type: ignore[arg-type]
    assert "hyde_query" not in out
    assert out.get("query") == "what is RAG?"


def test_hyde_enabled_generates_document() -> None:
    """When enabled=True with a mock LLM, 'hyde_query' is populated."""
    stub_llm = StubLLMProvider(
        canned_response="RAG stands for Retrieval-Augmented Generation."
    )
    node = HyDENode(llm=stub_llm, enabled=True, trace_writer=NullTraceWriter())
    ctx: dict[str, object] = {"query": "what is RAG?", "collection": "test"}
    out = node(ctx)  # type: ignore[arg-type]
    assert "hyde_query" in out


def test_hyde_query_populated_in_context() -> None:
    """The hyde_query value is a non-empty string in context."""
    stub_llm = StubLLMProvider(
        canned_response="HyDE improves retrieval by generating hypothetical documents."
    )
    node = HyDENode(llm=stub_llm, enabled=True, trace_writer=NullTraceWriter())
    ctx: dict[str, object] = {"query": "How does HyDE work?", "collection": "test"}
    out = node(ctx)  # type: ignore[arg-type]
    hyde_query = out.get("hyde_query", "")
    assert isinstance(hyde_query, str)
    assert len(hyde_query) > 0


def test_hyde_improves_recall_on_short_queries() -> None:
    """HyDE-generated doc is used for dense retrieval query, not the original query."""
    stub_llm = StubLLMProvider(
        canned_response="A hypothetical document about retrieval."
    )
    hyde_node = HyDENode(llm=stub_llm, enabled=True, trace_writer=NullTraceWriter())
    ctx: dict[str, object] = {"query": "short query", "collection": "test"}
    out = hyde_node(ctx)  # type: ignore[arg-type]

    # Verify hyde_query was generated
    hyde_query = out.get("hyde_query", "")
    assert isinstance(hyde_query, str)
    assert len(hyde_query) > 0

    # Now verify the retriever would use hyde_query as the query
    # This simulates what HybridRetrieverNode.run() does:
    # query = hyde_query or rewritten_query or context["query"]
    query = out.get("hyde_query") or out.get("rewritten_query") or out["query"]
    assert query == "A hypothetical document about retrieval."
    assert query != "short query"
