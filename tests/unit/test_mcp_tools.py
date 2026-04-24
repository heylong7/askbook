"""Unit tests for MCP tool handlers (DEV_SPEC Ch 30.1.3 — Task 3).

Uses StubEmbedder and real ChromaVectorStore backed by tmp_path (no mocks for
the vector store), plus a stub QueryPipeline for the ask handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import Chunk
from askbook.embeddings.stub import StubEmbedder
from askbook.providers.stub import StubLLMProvider
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.pipeline import QueryPipeline
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode
from askbook.rerankers.stub import StubReranker
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _build_pipeline(
    store: ChromaVectorStore,
    embedder: StubEmbedder,
    tmp_path: Path,
    *,
    llm_response: str = "This answers the question based on [c0]",
) -> QueryPipeline:
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    llm = StubLLMProvider(canned_response=llm_response)
    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    trace = _make_trace()
    return QueryPipeline(
        rewriter=QueryRewriterNode(enabled=False, trace_writer=trace),
        hyde=HyDENode(enabled=False, trace_writer=trace),
        retriever_node=HybridRetrieverNode(
            retriever=retriever, top_k=5, trace_writer=trace
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=trace),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(), top_k=3, trace_writer=trace
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, trace_writer=trace),
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=trace),
    )


def _seed_chunks(
    store: ChromaVectorStore,
    embedder: StubEmbedder,
    collection: str,
    *,
    doc_id: str = "doc_A",
    contents: list[str] | None = None,
) -> list[Chunk]:
    if contents is None:
        contents = ["alpha beta", "gamma delta", "epsilon zeta"]
    chunks = [
        Chunk(
            chunk_id=f"{doc_id}-c{i}",
            doc_id=doc_id,
            content=text,
            embedding=embedder.embed_passage(text),
            metadata={"source_path": f"/docs/{doc_id}.md"},
        )
        for i, text in enumerate(contents)
    ]
    store.upsert(chunks, collection)
    return chunks


def _build_deps(
    tmp_path: Path,
    *,
    seeded: bool = True,
    llm_response: str = "This answers the question based on [c0]",
    namespace: str = "demo",
) -> tuple[Any, StubEmbedder, ChromaVectorStore, str]:
    """Returns (ServerDeps, embedder, store, collection_name)."""
    from askbook.mcp_server.tools import ServerDeps
    from askbook.query.synthesizer import AnswerSynthesizerNode

    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name(
        namespace=namespace, embed_model=embedder.model_name
    )

    if seeded:
        _seed_chunks(store, embedder, collection)

    pipeline = _build_pipeline(store, embedder, tmp_path, llm_response=llm_response)
    deps = ServerDeps(
        pipeline=pipeline,
        store=store,
        embedder=embedder,
        fallback_text=AnswerSynthesizerNode.FALLBACK_TEXT,
    )
    return deps, embedder, store, collection


# ---------------------------------------------------------------------------
# search handler
# ---------------------------------------------------------------------------


def test_search_handler_returns_snippets_with_source_ids(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import SearchInput
    from askbook.mcp_server.tools import handle_search

    deps, _, _, _ = _build_deps(tmp_path)
    resp = handle_search(
        SearchInput(query="alpha", collection="demo", top_k=2), deps=deps
    )

    assert resp.status == "success"
    assert len(resp.source_ids) >= 1
    assert "hits" in resp.data
    assert all(len(h["snippet"]) <= 200 for h in resp.data["hits"])


def test_search_handler_warning_on_empty_collection(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import SearchInput
    from askbook.mcp_server.tools import handle_search

    # Empty store — no chunks seeded
    deps, _, _, _ = _build_deps(tmp_path, seeded=False)
    resp = handle_search(
        SearchInput(query="alpha", collection="demo", top_k=2), deps=deps
    )

    assert resp.status == "warning"
    assert resp.source_ids == []


# ---------------------------------------------------------------------------
# ask handler
# ---------------------------------------------------------------------------


def test_ask_handler_returns_answer_with_citations(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import AskInput
    from askbook.mcp_server.tools import handle_ask

    deps, _, _, _ = _build_deps(tmp_path, llm_response="Answer with citations [c0]")
    resp = handle_ask(AskInput(question="what is alpha?", collection="demo"), deps=deps)

    assert resp.status == "success"
    assert resp.data["answer"]  # non-empty string
    assert len(resp.source_ids) >= 1


def test_ask_handler_warning_on_fallback_answer(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import AskInput
    from askbook.mcp_server.tools import handle_ask

    # Empty retrieval → QueryPipeline returns FALLBACK_TEXT → handler returns warning
    deps, _, _, _ = _build_deps(tmp_path, seeded=False)
    resp = handle_ask(AskInput(question="what is alpha?", collection="demo"), deps=deps)

    assert resp.status == "warning"
    assert resp.source_ids == []


# ---------------------------------------------------------------------------
# list_collections handler
# ---------------------------------------------------------------------------


def test_list_collections_handler_returns_descriptors(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import ListCollectionsInput
    from askbook.mcp_server.tools import handle_list_collections

    deps, _, _, collection = _build_deps(tmp_path)
    resp = handle_list_collections(ListCollectionsInput(), deps=deps)

    assert resp.status == "success"
    # source_ids should contain the full collection name(s)
    assert len(resp.source_ids) >= 1
    assert collection in resp.source_ids
    assert "collections" in resp.data


def test_list_collections_handler_warning_when_empty(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import ListCollectionsInput
    from askbook.mcp_server.tools import handle_list_collections

    # Empty store — no upsert ever called
    deps, _, _, _ = _build_deps(tmp_path, seeded=False)
    resp = handle_list_collections(ListCollectionsInput(), deps=deps)

    assert resp.status == "warning"
    assert resp.source_ids == []


# ---------------------------------------------------------------------------
# get_document_summary handler
# ---------------------------------------------------------------------------


def test_get_document_summary_returns_first_snippets(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import GetDocumentSummaryInput
    from askbook.mcp_server.tools import handle_get_document_summary

    # Seed doc_A with 5 chunks
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name(
        namespace="demo", embed_model=embedder.model_name
    )
    _seed_chunks(
        store,
        embedder,
        collection,
        doc_id="doc_A",
        contents=[f"sentence {i}" for i in range(5)],
    )

    pipeline = _build_pipeline(store, embedder, tmp_path)
    from askbook.mcp_server.tools import ServerDeps
    from askbook.query.synthesizer import AnswerSynthesizerNode

    deps = ServerDeps(
        pipeline=pipeline,
        store=store,
        embedder=embedder,
        fallback_text=AnswerSynthesizerNode.FALLBACK_TEXT,
    )

    resp = handle_get_document_summary(
        GetDocumentSummaryInput(doc_id="doc_A", collection="demo"), deps=deps
    )

    assert resp.status == "success"
    assert len(resp.source_ids) >= 1
    first_snippets = resp.data["first_snippets"]
    assert len(first_snippets) <= 3
    assert all(len(s) <= 200 for s in first_snippets)


def test_get_document_summary_warning_when_doc_missing(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import GetDocumentSummaryInput
    from askbook.mcp_server.tools import handle_get_document_summary

    deps, _, _, _ = _build_deps(tmp_path)
    resp = handle_get_document_summary(
        GetDocumentSummaryInput(doc_id="nonexistent_doc_xyz", collection="demo"),
        deps=deps,
    )

    assert resp.status == "warning"
    assert resp.source_ids == []


# ---------------------------------------------------------------------------
# no raw_text keys
# ---------------------------------------------------------------------------


def test_handlers_never_emit_raw_text_key_in_hits(tmp_path: Path) -> None:
    from askbook.mcp_server.contracts import SearchInput
    from askbook.mcp_server.tools import handle_search

    deps, _, _, _ = _build_deps(tmp_path)
    resp = handle_search(
        SearchInput(query="alpha", collection="demo", top_k=5), deps=deps
    )

    for hit in resp.data.get("hits", []):
        assert "raw_text" not in hit
        assert "full_content" not in hit
        assert "page_content" not in hit


# ---------------------------------------------------------------------------
# TOOL_REGISTRY
# ---------------------------------------------------------------------------


def test_tool_registry_size_is_four_phase3_cap() -> None:
    from askbook.mcp_server.tools import TOOL_REGISTRY

    assert set(TOOL_REGISTRY) == {
        "search",
        "ask",
        "list_collections",
        "get_document_summary",
    }
    assert len(TOOL_REGISTRY) == 4
