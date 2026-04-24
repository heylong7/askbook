"""Harness 30.1.iii: every node.run() must be a pure function.

Same input always produces the same output.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import RetrievalResult
from askbook.providers.stub import StubLLMProvider
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode
from askbook.rerankers.stub import StubReranker


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _rrs(n: int = 3) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=str(i),
            score=float(i + 1),
            snippet="s",
            metadata={},
            retrieval_method="rrf",
        )
        for i in range(n)
    ]


BASE_CTX: dict[str, Any] = {
    "query": "what is RAG",
    "collection": "test",
    "pipeline_trace_id": "tid1",
    "rewritten_query": "what is RAG",
    "bm25_results": _rrs(),
    "dense_results": _rrs(),
    "retrieval_results": _rrs(),
}


def test_query_rewriter_node_idempotent() -> None:
    node = QueryRewriterNode(enabled=False, trace_writer=_make_trace())
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert out1["rewritten_query"] == out2["rewritten_query"]


def test_hyde_node_idempotent() -> None:
    node = HyDENode(enabled=False, trace_writer=_make_trace())
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert out1 == out2


def test_rrf_fusion_node_idempotent() -> None:
    node = RRFFusionNode(k=60, top_k=5, trace_writer=_make_trace())
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert [r.chunk_id for r in out1["retrieval_results"]] == [
        r.chunk_id for r in out2["retrieval_results"]
    ]


def test_cross_encoder_rerank_node_idempotent() -> None:
    node = CrossEncoderRerankNode(
        reranker=StubReranker(), top_k=3, trace_writer=_make_trace()
    )
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert [r.chunk_id for r in out1["retrieval_results"]] == [
        r.chunk_id for r in out2["retrieval_results"]
    ]


def test_llm_fine_rerank_node_idempotent() -> None:
    node = LLMFineRerankNode(enabled=False, trace_writer=_make_trace())
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert out1 == out2


def test_synthesizer_node_idempotent() -> None:
    node = AnswerSynthesizerNode(
        llm=StubLLMProvider(canned_response="ans"), trace_writer=_make_trace()
    )
    ctx = copy.deepcopy(BASE_CTX)
    out1 = node.run(ctx)  # type: ignore[arg-type]
    out2 = node.run(ctx)  # type: ignore[arg-type]
    assert out1["answer"].text == out2["answer"].text
    assert [c.chunk_id for c in out1["answer"].citations] == [
        c.chunk_id for c in out2["answer"].citations
    ]


def test_nodes_do_not_mutate_input_context() -> None:
    node = QueryRewriterNode(enabled=False, trace_writer=_make_trace())
    ctx = copy.deepcopy(BASE_CTX)
    ctx_before = copy.deepcopy(ctx)
    node.run(ctx)  # type: ignore[arg-type]
    # original should not be mutated
    assert ctx["query"] == ctx_before["query"]
    assert ctx.get("rewritten_query") == ctx_before.get("rewritten_query")
