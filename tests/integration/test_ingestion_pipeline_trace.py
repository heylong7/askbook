"""Integration tests: IngestionPipeline emits correct trace events via AsyncTraceWriter.

TDD Step 9.1 — RED: these tests must fail before the wiring is in place.
"""

from __future__ import annotations

import json
from pathlib import Path

from askbook.config.schema import ObservabilityConfig
from askbook.embeddings.stub import StubEmbedder
from askbook.ingestion.pipeline import IngestionPipeline
from askbook.observability.registry import build_trace_writer
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore


def _make_cfg(tmp_path: Path) -> ObservabilityConfig:
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


def _build_pipeline(tmp_path: Path, writer: object) -> IngestionPipeline:
    data = tmp_path / "data"
    store = ChromaVectorStore(path=str(data / "chroma"))
    bm25 = BM25PersistentIndex(path=data / "bm25" / "demo.pkl")
    return IngestionPipeline(
        embedder=StubEmbedder(dimension=8),
        store=store,
        bm25_index=bm25,
        chunk_size=80,
        chunk_overlap=20,
        trace_writer=writer,  # type: ignore[arg-type]
    )


def _make_source(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "sample.md").write_text(
        "# Sample\n\n" + "Hello world. " * 50,
        encoding="utf-8",
    )
    return docs


def test_ingestion_emits_spans_for_all_nodes(tmp_path: Path) -> None:
    """Running the pipeline must produce span_end events for all major nodes."""
    cfg = _make_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    pipeline = _build_pipeline(tmp_path, writer)
    source = _make_source(tmp_path)

    pipeline.run(source=source, collection="demo")
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = [e for e in events if e["event_type"] == "span_end"]
    node_names = {e["node_name"] for e in span_ends}

    required_nodes = {"load", "split", "enrich", "dedup", "embed", "write", "bm25"}
    missing = required_nodes - node_names
    assert not missing, (
        f"Missing span_end events for nodes: {missing}. Got node_names: {node_names}"
    )


def test_ingestion_span_includes_counts(tmp_path: Path) -> None:
    """Span tags must include chunk_count (split) and written_count (write)."""
    cfg = _make_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    pipeline = _build_pipeline(tmp_path, writer)
    source = _make_source(tmp_path)

    pipeline.run(source=source, collection="demo")
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = {e["node_name"]: e for e in events if e["event_type"] == "span_end"}

    # split node must report chunk_count >= 1
    assert "split" in span_ends, "No split span_end found"
    split_tags = span_ends["split"].get("tags", {})
    assert "chunk_count" in split_tags, (
        f"split span missing chunk_count tag; got: {split_tags}"
    )
    assert split_tags["chunk_count"] >= 1, (
        f"Expected chunk_count >= 1, got {split_tags['chunk_count']}"
    )

    # write node must report written_count >= 1
    assert "write" in span_ends, "No write span_end found"
    write_tags = span_ends["write"].get("tags", {})
    assert "written_count" in write_tags, (
        f"write span missing written_count tag; got: {write_tags}"
    )
    assert write_tags["written_count"] >= 1, (
        f"Expected written_count >= 1, got {write_tags['written_count']}"
    )


def test_ingestion_trace_id_stable_across_all_nodes(tmp_path: Path) -> None:
    """All span_end events from one pipeline.run() must share the same trace_id."""
    cfg = _make_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    pipeline = _build_pipeline(tmp_path, writer)
    source = _make_source(tmp_path)

    pipeline.run(source=source, collection="demo")
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = [e for e in events if e["event_type"] == "span_end"]

    assert len(span_ends) >= 7, (
        f"Expected >= 7 span_end events (all pipeline nodes), got {len(span_ends)}"
    )

    trace_ids = {e["trace_id"] for e in span_ends}
    assert len(trace_ids) == 1, f"All spans must share one trace_id, got: {trace_ids}"


def test_ingestion_dry_run_still_emits_trace(tmp_path: Path) -> None:
    """In dry-run mode, trace events must still be written for the nodes that ran."""
    cfg = _make_cfg(tmp_path)
    writer = build_trace_writer(cfg)
    pipeline = _build_pipeline(tmp_path, writer)
    source = _make_source(tmp_path)

    pipeline.run(source=source, collection="demo", dry_run=True)
    writer.flush()

    trace_dir = tmp_path / "traces"
    events = _read_events(trace_dir)
    span_ends = {e["node_name"] for e in events if e["event_type"] == "span_end"}

    # Nodes that run in dry-run mode
    assert "load" in span_ends
    assert "split" in span_ends
    assert "enrich" in span_ends

    # Nodes that are skipped in dry-run mode — must NOT appear
    assert "write" not in span_ends, (
        f"write node should not run in dry-run mode, but got span_ends: {span_ends}"
    )
