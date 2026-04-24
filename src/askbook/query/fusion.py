"""Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from __future__ import annotations

from askbook.core.interfaces import (
    BasePipelineNode,
    PipelineContext,
    TraceWriterProtocol,
)
from askbook.core.models import RetrievalResult


def rrf_fusion(
    result_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """score(d) = sum(1 / (k + rank)) across all input lists.

    Preserves first-seen snippet & metadata; method fixed to 'rrf'.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}
    for lst in result_lists:
        for rank, r in enumerate(lst, 1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(r.chunk_id, r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[RetrievalResult] = []
    for cid, s in ordered:
        src = first_seen[cid]
        out.append(
            RetrievalResult(
                chunk_id=cid,
                score=float(s),
                snippet=src.snippet[:200],
                metadata=dict(src.metadata),
                retrieval_method="rrf",
            )
        )
    return out[:top_k] if top_k else out


class RRFFusionNode(BasePipelineNode):
    def __init__(
        self, *, k: int, top_k: int, trace_writer: TraceWriterProtocol
    ) -> None:
        super().__init__(name="rrf_fusion", trace_writer=trace_writer)
        self._k = k
        self._top_k = top_k

    def run(self, context: PipelineContext) -> PipelineContext:
        bm25 = context.get("bm25_results", [])
        dense = context.get("dense_results", [])
        fused = rrf_fusion([bm25, dense], k=self._k, top_k=self._top_k)
        return {**context, "retrieval_results": fused}


__all__ = ["rrf_fusion", "RRFFusionNode"]
