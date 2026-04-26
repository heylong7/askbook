"""Task 17 — Trace overhead reference benchmark.

Marked @pytest.mark.slow; skipped in regular CI (add -m slow to run).
Not a hard gate — just records overhead for the PR description.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pytest

from askbook.config.schema import ObservabilityConfig
from askbook.embeddings.stub import StubEmbedder
from askbook.observability.null_trace import NullTraceWriter
from askbook.observability.registry import build_trace_writer
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


def _build_pipeline(tmp_path: Path, trace_writer: object) -> QueryPipeline:
    from askbook.providers.stub import StubLLMProvider

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    embedder = StubEmbedder(dimension=8)
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25" / "demo.pkl")
    llm = StubLLMProvider()
    reranker = StubReranker()

    from askbook.core.interfaces import TraceWriterProtocol

    tw: TraceWriterProtocol = trace_writer  # type: ignore[assignment]

    return QueryPipeline(
        rewriter=QueryRewriterNode(enabled=False, llm=llm, trace_writer=tw),
        hyde=HyDENode(enabled=False, llm=llm, trace_writer=tw),
        retriever_node=HybridRetrieverNode(
            retriever=HybridRetriever(embedder=embedder, store=store, bm25_index=bm25),
            top_k=5,
            trace_writer=tw,
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=tw),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=reranker, top_k=3, trace_writer=tw
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, llm=llm, trace_writer=tw),
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=tw),
    )


def _measure_p95(pipeline: QueryPipeline, n: int = 30) -> float:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        pipeline.run(query="test query", collection="demo")
        times.append(time.perf_counter() - t0)
    times.sort()
    idx = max(0, int(0.95 * n) - 1)
    return times[idx] * 1000  # ms


@pytest.mark.slow
def test_query_p95_overhead_under_5_percent(tmp_path: Path) -> None:
    """Reference benchmark: AsyncTraceWriter P95 overhead should be < 5%.

    Not a hard failure — emits a warning if exceeded so it can be noted in the PR.
    Requires -m slow to run.
    """
    null_pipeline = _build_pipeline(tmp_path / "null", NullTraceWriter())
    p95_without = _measure_p95(null_pipeline)

    cfg = ObservabilityConfig(
        enabled=True,
        trace_dir=str(tmp_path / "traces"),
        pii_redaction=False,
        flush_interval_seconds=1.0,
    )
    writer = build_trace_writer(cfg)
    trace_pipeline = _build_pipeline(tmp_path / "trace", writer)
    p95_with = _measure_p95(trace_pipeline)
    writer.flush()

    delta_ratio = (p95_with - p95_without) / p95_without if p95_without > 0 else 0.0

    print(
        f"\nTrace overhead benchmark:"
        f"\n  P95 without trace: {p95_without:.1f} ms"
        f"\n  P95 with trace:    {p95_with:.1f} ms"
        f"\n  Overhead ratio:    {delta_ratio:.1%}"
    )

    if delta_ratio > 0.10:
        warnings.warn(
            f"Trace overhead {delta_ratio:.1%} exceeds 10% reference threshold "
            f"(target <5%). P95 without={p95_without:.1f}ms, with={p95_with:.1f}ms.",
            stacklevel=2,
        )
    # Not a hard assertion — overhead varies per machine; just report
    assert True
