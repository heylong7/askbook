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
        self._model_name = model
        self._use_fp16 = use_fp16
        self._reranker: object | None = None  # lazy

    def _get_reranker(self) -> object:
        if self._reranker is None:
            from FlagEmbedding import FlagReranker  # lazy import

            self._reranker = FlagReranker(self._model_name, use_fp16=self._use_fp16)
        return self._reranker

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        reranker = self._get_reranker()
        pairs = [(query, r.snippet) for r in results]
        scores = reranker.compute_score(pairs, normalize=True)  # type: ignore[attr-defined]
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
