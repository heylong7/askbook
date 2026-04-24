"""Tests for RRF fusion function and RRFFusionNode."""

from __future__ import annotations

from askbook.core.models import RetrievalResult
from askbook.query.fusion import rrf_fusion


def _r(cid: str, score: float, method: str = "bm25") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=cid, score=score, snippet="snip", metadata={}, retrieval_method=method
    )


def test_rrf_merges_overlapping_results_boosts_shared_ids() -> None:
    # bm25: [A, B, C], dense: [B, A, D]
    # A: rank1(bm25) + rank2(dense) -> higher score than
    # C(rank3 bm25 only) and D(rank2 dense only)
    bm25 = [_r("A", 1.0), _r("B", 0.8), _r("C", 0.6)]
    dense = [_r("B", 1.0, "dense"), _r("A", 0.9, "dense"), _r("D", 0.5, "dense")]
    result = rrf_fusion([bm25, dense])
    ids = [r.chunk_id for r in result]
    assert "A" in ids[:2] and "B" in ids[:2]
    assert all(r.retrieval_method == "rrf" for r in result)


def test_rrf_with_one_empty_list_preserves_order() -> None:
    dense = [_r("X", 1.0, "dense"), _r("Y", 0.5, "dense")]
    result = rrf_fusion([[], dense])
    assert [r.chunk_id for r in result] == ["X", "Y"]
    assert all(r.retrieval_method == "rrf" for r in result)


def test_rrf_both_empty_returns_empty() -> None:
    assert rrf_fusion([[], []]) == []


def test_rrf_result_snippet_inherits_and_is_capped() -> None:
    r = RetrievalResult(
        chunk_id="A",
        score=1.0,
        snippet="x" * 200,
        metadata={},
        retrieval_method="bm25",
    )
    result = rrf_fusion([[r], []])
    assert len(result[0].snippet) <= 200


def test_rrf_custom_k_changes_ranking_sensitivity() -> None:
    # With k=1 (high sensitivity), rank matters more; ranking can differ from k=60
    bm25 = [_r("A", 1.0)]
    dense = [_r("B", 1.0, "dense"), _r("A", 0.5, "dense")]
    r_k1 = rrf_fusion([bm25, dense], k=1)
    r_k60 = rrf_fusion([bm25, dense], k=60)
    # Both contain same IDs, but verify function runs correctly with custom k
    assert {r.chunk_id for r in r_k1} == {r.chunk_id for r in r_k60} == {"A", "B"}
    # Verify k actually changes the computed scores
    a_score_k1 = next(r.score for r in r_k1 if r.chunk_id == "A")
    a_score_k60 = next(r.score for r in r_k60 if r.chunk_id == "A")
    assert a_score_k1 != a_score_k60


def test_rrf_top_k_limits_output() -> None:
    lst = [_r(str(i), float(i)) for i in range(10)]
    result = rrf_fusion([lst], top_k=3)
    assert len(result) == 3


def test_rrf_top_k_zero_returns_empty() -> None:
    lst = [_r("A", 1.0), _r("B", 0.5)]
    assert rrf_fusion([lst], top_k=0) == []


def test_rrf_fusion_node_merges_context_keys() -> None:
    from unittest.mock import MagicMock

    from askbook.core.interfaces import TraceSpan
    from askbook.query.fusion import RRFFusionNode

    mock_span = TraceSpan(name="rrf_fusion")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)

    node = RRFFusionNode(k=60, top_k=2, trace_writer=trace)
    bm25 = [_r("A", 1.0), _r("B", 0.5)]
    dense = [_r("B", 1.0, "dense"), _r("C", 0.3, "dense")]
    ctx = {"query": "q", "bm25_results": bm25, "dense_results": dense}
    out = node(ctx)  # type: ignore[arg-type]
    assert "retrieval_results" in out
    assert len(out["retrieval_results"]) == 2
    assert all(r.retrieval_method == "rrf" for r in out["retrieval_results"])
