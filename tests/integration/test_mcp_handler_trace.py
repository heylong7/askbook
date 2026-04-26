"""Integration tests: MCP handler functions emit trace spans via AsyncTraceWriter.

TDD Step 10 — RED: these tests must fail before the wiring is in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from askbook.config.schema import ObservabilityConfig
from askbook.core.models import Chunk
from askbook.embeddings.stub import StubEmbedder
from askbook.mcp_server.contracts import AskInput
from askbook.mcp_server.tools import ServerDeps, handle_ask
from askbook.observability.registry import build_trace_writer
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


def _make_obs_cfg(tmp_path: Path) -> ObservabilityConfig:
    return ObservabilityConfig(
        enabled=True,
        trace_dir=str(tmp_path / "traces"),
        pii_redaction=False,
        flush_interval_seconds=60.0,  # don't auto-flush; we flush manually
    )


def _read_events(trace_dir: Path) -> list[dict]:
    events: list[dict] = []
    for p in sorted(trace_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _build_pipeline(
    store: ChromaVectorStore,
    embedder: StubEmbedder,
    tmp_path: Path,
    *,
    llm_response: str = "This answers the question based on [c0]",
) -> QueryPipeline:
    from unittest.mock import MagicMock

    from askbook.core.interfaces import TraceSpan

    def _make_null_trace() -> object:
        span = TraceSpan(name="test")
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=span)
        cm.__exit__ = MagicMock(return_value=False)
        trace = MagicMock()
        trace.span = MagicMock(return_value=cm)
        return trace

    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    llm = StubLLMProvider(canned_response=llm_response)
    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    null_trace = _make_null_trace()
    return QueryPipeline(
        rewriter=QueryRewriterNode(enabled=False, trace_writer=null_trace),  # type: ignore[arg-type]
        hyde=HyDENode(enabled=False, trace_writer=null_trace),  # type: ignore[arg-type]
        retriever_node=HybridRetrieverNode(
            retriever=retriever,
            top_k=5,
            trace_writer=null_trace,  # type: ignore[arg-type]
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=null_trace),  # type: ignore[arg-type]
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(),
            top_k=3,
            trace_writer=null_trace,  # type: ignore[arg-type]
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, trace_writer=null_trace),  # type: ignore[arg-type]
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=null_trace),  # type: ignore[arg-type]
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


def _make_deps(
    tmp_path: Path,
    writer: object,
    *,
    seeded: bool = True,
    llm_response: str = "This answers the question based on [c0]",
) -> ServerDeps:
    embedder = StubEmbedder(dimension=8)
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name(
        namespace="demo", embed_model=embedder.model_name
    )
    if seeded:
        _seed_chunks(store, embedder, collection)

    pipeline = _build_pipeline(store, embedder, tmp_path, llm_response=llm_response)
    return ServerDeps(
        pipeline=pipeline,
        store=store,
        embedder=embedder,
        fallback_text=AnswerSynthesizerNode.FALLBACK_TEXT,
        trace_writer=writer,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_handle_ask_emits_mcp_tool_span(tmp_path: Path) -> None:
    """handle_ask must emit a span_end event with node_name='mcp.ask'."""
    cfg = _make_obs_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    deps = _make_deps(tmp_path, writer)

    handle_ask(AskInput(question="what is alpha?", collection="demo"), deps=deps)
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = [e for e in events if e["event_type"] == "span_end"]
    mcp_spans = [e for e in span_ends if e["node_name"] == "mcp.ask"]

    node_names = [e["node_name"] for e in span_ends]
    assert mcp_spans, f"No mcp.ask span_end found. Got node_names: {node_names}"
    mcp_span = mcp_spans[0]
    tags = mcp_span.get("tags", {})
    assert tags.get("status") in ("success", "warning"), (
        f"Expected status 'success' or 'warning', got: {tags.get('status')}"
    )


def test_handle_ask_warning_when_retrieval_empty(tmp_path: Path) -> None:
    """handle_ask with empty store emits mcp.ask span_end with status='warning'."""
    cfg = _make_obs_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    deps = _make_deps(tmp_path, writer, seeded=False)

    resp = handle_ask(AskInput(question="what is alpha?", collection="demo"), deps=deps)
    writer.flush()

    assert resp.status == "warning"

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = [e for e in events if e["event_type"] == "span_end"]
    mcp_spans = [e for e in span_ends if e["node_name"] == "mcp.ask"]

    assert mcp_spans, (
        f"No mcp.ask span_end found. Got: {[e['node_name'] for e in span_ends]}"
    )
    tags = mcp_spans[0].get("tags", {})
    assert tags.get("status") == "warning", (
        f"Expected status='warning', got: {tags.get('status')}"
    )


def test_mcp_span_parent_of_query_pipeline_spans(tmp_path: Path) -> None:
    """The mcp.ask span must be the parent of all query pipeline span_end events.

    The mcp.ask span opens first; the query pipeline runs inside it, so all
    pipeline node spans should have parent_span_id == mcp.ask span_id.

    Note: QueryPipeline calls use_trace_id() which resets the trace_id contextvar,
    but the span_stack contextvar retains the mcp.ask span_id on the stack so
    child spans pick up the correct parent_span_id.
    """
    cfg = _make_obs_cfg(tmp_path)
    writer = build_trace_writer(cfg)

    # Build deps where the pipeline nodes also use the same writer
    embedder = StubEmbedder(dimension=8)
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name(
        namespace="demo", embed_model=embedder.model_name
    )
    _seed_chunks(store, embedder, collection)

    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    llm = StubLLMProvider(canned_response="This answers the question based on [c0]")
    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    pipeline = QueryPipeline(
        rewriter=QueryRewriterNode(enabled=False, trace_writer=writer),
        hyde=HyDENode(enabled=False, trace_writer=writer),
        retriever_node=HybridRetrieverNode(
            retriever=retriever, top_k=5, trace_writer=writer
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=writer),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(), top_k=3, trace_writer=writer
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, trace_writer=writer),
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=writer),
    )
    deps = ServerDeps(
        pipeline=pipeline,
        store=store,
        embedder=embedder,
        fallback_text=AnswerSynthesizerNode.FALLBACK_TEXT,
        trace_writer=writer,
    )

    handle_ask(AskInput(question="what is alpha?", collection="demo"), deps=deps)
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = [e for e in events if e["event_type"] == "span_end"]

    # Find mcp.ask span
    mcp_spans = [e for e in span_ends if e["node_name"] == "mcp.ask"]
    assert mcp_spans, (
        f"No mcp.ask span_end found. Got: {[e['node_name'] for e in span_ends]}"
    )
    mcp_span = mcp_spans[0]
    mcp_span_id = mcp_span["span_id"]

    # Find pipeline node spans (not the mcp.ask span itself)
    pipeline_spans = [e for e in span_ends if e["node_name"] != "mcp.ask"]
    assert pipeline_spans, "Expected at least one pipeline node span_end"

    # All pipeline spans should have parent_span_id == mcp.ask span_id
    for ps in pipeline_spans:
        assert ps.get("parent_span_id") == mcp_span_id, (
            f"Pipeline span '{ps['node_name']}' has parent_span_id="
            f"{ps.get('parent_span_id')!r}, expected {mcp_span_id!r}"
        )
