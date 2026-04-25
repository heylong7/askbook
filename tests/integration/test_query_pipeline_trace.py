"""Integration tests: QueryPipeline emits correct trace events via AsyncTraceWriter.

TDD Step 8.1 — RED: these tests must fail before the wiring is in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from askbook.config.schema import ObservabilityConfig
from askbook.embeddings.stub import StubEmbedder
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


def _read_events(tmp_path: Path) -> list[dict]:
    events: list[dict] = []
    for p in sorted(tmp_path.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _build_pipeline(
    tmp_path: Path, *, writer: object, rewrite: bool = False
) -> QueryPipeline:
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    llm = StubLLMProvider(canned_response="stub answer [c0]")
    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    return QueryPipeline(
        rewriter=QueryRewriterNode(enabled=rewrite, llm=llm, trace_writer=writer),  # type: ignore[arg-type]
        hyde=HyDENode(enabled=False, llm=llm, trace_writer=writer),  # type: ignore[arg-type]
        retriever_node=HybridRetrieverNode(
            retriever=retriever,
            top_k=5,
            trace_writer=writer,  # type: ignore[arg-type]
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=writer),  # type: ignore[arg-type]
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(),
            top_k=3,
            trace_writer=writer,  # type: ignore[arg-type]
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, trace_writer=writer),  # type: ignore[arg-type]
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=writer),  # type: ignore[arg-type]
    )


def _make_cfg(tmp_path: Path) -> ObservabilityConfig:
    return ObservabilityConfig(
        enabled=True,
        trace_dir=str(tmp_path),
        pii_redaction=False,
        flush_interval_seconds=60.0,  # don't auto-flush; we flush manually
    )


def test_query_emits_at_least_seven_spans(tmp_path: Path) -> None:
    """Running the 7-node pipeline must produce ≥ 7 span_end events."""
    writer = build_trace_writer(_make_cfg(tmp_path))
    pipeline = _build_pipeline(tmp_path, writer=writer)
    pipeline.run(query="什么是 RRF？", collection="demo")
    writer.flush()

    events = _read_events(tmp_path)
    span_ends = [e for e in events if e["event_type"] == "span_end"]
    node_names = [e["node_name"] for e in span_ends]
    assert len(span_ends) >= 7, (
        f"Expected ≥ 7 span_end events, got {len(span_ends)}: {node_names}"
    )


def test_query_span_records_original_query_when_rewrite_disabled(
    tmp_path: Path,
) -> None:
    """When rewrite is disabled, rewriter span_end must carry original_query in tags."""
    writer = build_trace_writer(_make_cfg(tmp_path))
    pipeline = _build_pipeline(tmp_path, writer=writer, rewrite=False)
    query = "什么是 RRF？"
    pipeline.run(query=query, collection="demo")
    writer.flush()

    events = _read_events(tmp_path)
    rewriter_ends = [
        e
        for e in events
        if e["event_type"] == "span_end" and e["node_name"] == "query_rewriter"
    ]
    assert len(rewriter_ends) >= 1, "No query_rewriter span_end found"
    tags = rewriter_ends[0].get("tags", {})
    assert tags.get("original_query") == query, (
        f"Expected original_query={query!r}, got tags={tags}"
    )


def test_synthesizer_span_carries_source_ids(tmp_path: Path) -> None:
    """The synthesizer span_end must have answer_source_ids tag (may be empty list)."""
    writer = build_trace_writer(_make_cfg(tmp_path))
    pipeline = _build_pipeline(tmp_path, writer=writer)
    pipeline.run(query="什么是 RRF？", collection="demo")
    writer.flush()

    events = _read_events(tmp_path)
    synth_ends = [
        e
        for e in events
        if e["event_type"] == "span_end" and e["node_name"] == "answer_synthesizer"
    ]
    assert len(synth_ends) >= 1, "No answer_synthesizer span_end found"
    tags = synth_ends[0].get("tags", {})
    assert "answer_source_ids" in tags, (
        f"Expected answer_source_ids in tags, got {tags}"
    )
    assert isinstance(tags["answer_source_ids"], list), (
        "answer_source_ids must be a list"
    )


def test_query_trace_id_stable_across_all_nodes(tmp_path: Path) -> None:
    """All span_end events from a single pipeline.run() must share the same trace_id."""
    writer = build_trace_writer(_make_cfg(tmp_path))
    pipeline = _build_pipeline(tmp_path, writer=writer)
    pipeline.run(query="什么是 RRF？", collection="demo")
    writer.flush()

    events = _read_events(tmp_path)
    span_ends = [e for e in events if e["event_type"] == "span_end"]
    assert len(span_ends) >= 7, f"Expected ≥ 7 span_end events, got {len(span_ends)}"

    trace_ids = {e["trace_id"] for e in span_ends}
    assert len(trace_ids) == 1, f"All spans must share one trace_id, got: {trace_ids}"


def test_query_trace_no_raw_text_in_tags(tmp_path: Path) -> None:
    """No trace event may contain raw_text, full_content, or page_content in tags."""
    writer = build_trace_writer(_make_cfg(tmp_path))
    pipeline = _build_pipeline(tmp_path, writer=writer)
    pipeline.run(query="什么是 RRF？", collection="demo")
    writer.flush()

    forbidden = {"raw_text", "full_content", "page_content"}
    events = _read_events(tmp_path)
    for event in events:
        tags = event.get("tags", {})
        found = forbidden & set(tags.keys())
        assert not found, f"Forbidden tag(s) {found} in event {event}"
