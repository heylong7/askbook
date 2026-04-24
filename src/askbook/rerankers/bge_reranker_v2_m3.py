"""BGE-Reranker-v2-m3: lazy-loads FlagEmbedding to avoid import cost in tests."""

from __future__ import annotations

import asyncio

from askbook.core.models import RetrievalResult


class BGERerankerV2M3:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3 (lazy import)."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import FlagReranker  # lazy import

        self._reranker = FlagReranker(model, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.snippet) for r in results]
        scores = self._reranker.compute_score(pairs, normalize=True)
        scored = sorted(
            zip(results, scores, strict=True),
            key=lambda rs: float(rs[1]),
            reverse=True,
        )
        return [
            r.model_copy(update={"score": float(s), "retrieval_method": "reranked"})
            for r, s in scored[:top_k]
        ]

    async def arerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        return await asyncio.to_thread(self.rerank, query, results, top_k)


__all__ = ["BGERerankerV2M3"]
