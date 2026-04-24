"""QueryPipeline: orchestrates the 7-node query chain."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from askbook.core.interfaces import PipelineContext
from askbook.core.models import Answer
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.retriever import HybridRetrieverNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode


@dataclass
class QueryPipeline:
    """Orchestrates the 7-node query chain.

    Order: Rewriter → HyDE → HybridRetriever → RRF → CrossEncoder
           → LLMRerank → Synthesizer.
    """

    rewriter: QueryRewriterNode
    hyde: HyDENode
    retriever_node: HybridRetrieverNode
    fusion_node: RRFFusionNode
    ce_rerank_node: CrossEncoderRerankNode
    llm_rerank_node: LLMFineRerankNode
    synthesizer_node: AnswerSynthesizerNode

    def run(self, *, query: str, collection: str) -> Answer:
        trace_id = uuid.uuid4().hex
        ctx: PipelineContext = {
            "query": query,
            "collection": collection,
            "pipeline_trace_id": trace_id,
        }
        for node in (
            self.rewriter,
            self.hyde,
            self.retriever_node,
            self.fusion_node,
            self.ce_rerank_node,
            self.llm_rerank_node,
            self.synthesizer_node,
        ):
            ctx = node(ctx)
        return ctx["answer"]


__all__ = ["QueryPipeline"]
