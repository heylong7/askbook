"""CrossEncoderRerankNode and LLMFineRerankNode pipeline stages."""

from __future__ import annotations

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    RerankerProtocol,
    TraceWriterProtocol,
)


class CrossEncoderRerankNode(BasePipelineNode):
    """Reranks retrieval_results using a cross-encoder (or StubReranker in tests)."""

    def __init__(
        self,
        *,
        reranker: RerankerProtocol,
        top_k: int,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="cross_encoder_rerank", trace_writer=trace_writer)
        self._reranker = reranker
        self._top_k = top_k

    def run(self, context: PipelineContext) -> PipelineContext:
        results = context.get("retrieval_results", [])
        if not results:
            return context
        query = context.get("rewritten_query") or context["query"]
        reranked = self._reranker.rerank(query, results, self._top_k)
        return {**context, "retrieval_results": reranked}


class LLMFineRerankNode(BasePipelineNode):
    """Disabled-by-default LLM-based rerank. v0.5 opt-in."""

    def __init__(
        self,
        *,
        llm: LLMProviderProtocol | None = None,
        enabled: bool = False,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="llm_fine_rerank", trace_writer=trace_writer)
        self._llm = llm
        self._enabled = enabled

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self._enabled or not context.get("retrieval_results"):
            return context
        # v0.5: use llm to score (query, snippet) pairs
        return context


__all__ = ["CrossEncoderRerankNode", "LLMFineRerankNode"]
