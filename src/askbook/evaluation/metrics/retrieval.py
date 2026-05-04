"""Retrieval-quality metrics -- pure functions, zero LLM cost (DEV_SPEC Ch 25)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _truncate(retrieved: Sequence[str], k: int) -> list[str]:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return list(retrieved[:k])


def hit_rate(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any of top-k retrieved is in relevant set; 0.0 otherwise."""
    if not relevant:
        return 0.0
    top = _truncate(retrieved, k)
    return 1.0 if any(rid in relevant for rid in top) else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item in top-k; 0.0 if none."""
    if not relevant:
        return 0.0
    for i, rid in enumerate(_truncate(retrieved, k), start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """|relevant n top-k| / |relevant|; 0.0 if relevant is empty."""
    if not relevant:
        return 0.0
    top = set(_truncate(retrieved, k))
    return len(top & relevant) / len(relevant)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: DCG / IDCG.

    DCG = sum_{i=1..k}  rel_i / log2(i+1)
    IDCG = sum_{i=1..min(|relevant|, k)}  1 / log2(i+1)
    """
    if not relevant:
        return 0.0
    top = _truncate(retrieved, k)
    dcg = sum(
        1.0 / math.log2(i + 1) for i, rid in enumerate(top, start=1) if rid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


__all__ = ["hit_rate", "mrr", "recall_at_k", "ndcg_at_k"]
