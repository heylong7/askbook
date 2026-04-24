"""StubReranker: deterministic reranker for tests (sorts by score desc)."""

from __future__ import annotations

from askbook.core.models import RetrievalResult


class StubReranker:
    """Sorts results by existing score descending — deterministic for tests."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        ordered = sorted(results, key=lambda r: r.score, reverse=True)
        return [
            r.model_copy(update={"retrieval_method": "reranked"})
            for r in ordered[:top_k]
        ]

    async def arerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        return self.rerank(query, results, top_k)


__all__ = ["StubReranker"]
