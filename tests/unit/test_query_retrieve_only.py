"""Tests for QueryPipeline.run_retrieve_only — evaluation short-circuit."""

from __future__ import annotations

from askbook.core.interfaces import BasePipelineNode, PipelineContext
from askbook.core.models import RetrievalResult
from askbook.observability.null_trace import NullTraceWriter
from askbook.rerankers.stub import StubReranker


class _FakeRetrieverNode(BasePipelineNode):
    """Returns fixed RetrievalResults — never fails."""

    def __init__(self) -> None:
        super().__init__(name="retriever", trace_writer=NullTraceWriter())

    def run(self, context: PipelineContext) -> PipelineContext:
        results = [
            RetrievalResult(
                chunk_id="c1", score=0.9, snippet="...", retrieval_method="bm25"
            ),
            RetrievalResult(
                chunk_id="c2", score=0.5, snippet="...", retrieval_method="bm25"
            ),
        ]
        return {
            **context,
            "bm25_results": results,
            "dense_results": results,
            "retrieval_results": results,
        }


class _FailingSynthesizer(BasePipelineNode):
    """Synthesizer that raises if called — must not be reached."""

    def __init__(self) -> None:
        super().__init__(name="synthesizer", trace_writer=NullTraceWriter())

    def run(self, context: PipelineContext) -> PipelineContext:
        raise RuntimeError("Synthesizer must not be called in retrieve-only mode")


def _make_pipeline():
    from askbook.query.fusion import RRFFusionNode
    from askbook.query.hyde import HyDENode
    from askbook.query.pipeline import QueryPipeline
    from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
    from askbook.query.rewriter import QueryRewriterNode

    # Use disabled nodes for rewriter/hyde/llm_rerank to avoid external deps
    return QueryPipeline(
        rewriter=QueryRewriterNode(
            enabled=False, llm=None, trace_writer=NullTraceWriter()
        ),
        hyde=HyDENode(enabled=False, llm=None, trace_writer=NullTraceWriter()),
        retriever_node=_FakeRetrieverNode(),
        fusion_node=RRFFusionNode(k=60, top_k=10, trace_writer=NullTraceWriter()),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(), top_k=5, trace_writer=NullTraceWriter()
        ),
        llm_rerank_node=LLMFineRerankNode(
            enabled=False, llm=None, trace_writer=NullTraceWriter()
        ),
        synthesizer_node=_FailingSynthesizer(),
    )


def test_run_retrieve_only_returns_retrieval_results():
    pipeline = _make_pipeline()
    results = pipeline.run_retrieve_only(query="test q", collection="demo")
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].chunk_id == "c1"


def test_run_retrieve_only_skips_synthesizer():
    pipeline = _make_pipeline()
    # _FailingSynthesizer would raise if called — no exception means success
    results = pipeline.run_retrieve_only(query="test q", collection="demo")
    assert len(results) == 2
