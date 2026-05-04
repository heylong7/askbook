"""Tests for RetrievalEvalReport — per-qid breakdown, aggregation, baseline diff."""

from __future__ import annotations

import pytest

from askbook.evaluation.report import (
    PerQueryRow,
    RetrievalEvalReport,
    compare_to_baseline,
)


def test_aggregate_means_correct_for_two_items():
    report = RetrievalEvalReport(
        dataset_path="d.yaml",
        collection="c",
        k=5,
        rows=[
            PerQueryRow(
                qid="Q1",
                metrics={"hit_rate": 1.0, "mrr": 1.0},
                latency_ms=10,
                retrieved_top_ids=["a", "b"],
            ),
            PerQueryRow(
                qid="Q2",
                metrics={"hit_rate": 0.0, "mrr": 0.5},
                latency_ms=20,
                retrieved_top_ids=["c", "d"],
            ),
        ],
    )
    agg = report.aggregate
    assert agg["hit_rate"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.75)


def test_format_table_contains_all_metric_columns():
    report = RetrievalEvalReport(
        dataset_path="d.yaml",
        collection="demo",
        k=5,
        rows=[
            PerQueryRow(
                qid="Q1",
                metrics={
                    "hit_rate": 1.0,
                    "mrr": 1.0,
                    "recall@5": 0.8,
                    "ndcg@5": 0.9,
                },
                latency_ms=12,
                retrieved_top_ids=["a"],
            ),
        ],
    )
    table = report.format_table()
    assert "hit_rate" in table
    assert "mrr" in table
    assert "recall@5" in table
    assert "ndcg@5" in table
    assert "latency_p50" in table
    assert "latency_p90" in table


def test_to_json_roundtrip_preserves_per_qid_breakdown():
    report = RetrievalEvalReport(
        dataset_path="d.yaml",
        collection="c",
        k=5,
        rows=[
            PerQueryRow(
                qid="Q1", metrics={"mrr": 1.0}, latency_ms=5, retrieved_top_ids=["x"]
            ),
        ],
    )
    payload = report.to_json()
    assert payload["dataset_path"] == "d.yaml"
    assert payload["k"] == 5
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["qid"] == "Q1"
    assert payload["rows"][0]["metrics"]["mrr"] == 1.0


def test_compare_to_baseline_flags_drop_above_threshold():
    baseline = {"recall@5": 0.80, "mrr": 0.70}
    current = {"recall@5": 0.74, "mrr": 0.68}
    failures = compare_to_baseline(current, baseline, threshold=0.05)
    # recall@5: drop=0.06 > 0.05 => flagged; mrr: drop=0.02 <= 0.05 => not flagged
    assert len(failures) == 1
    metric, base, cur, drop = failures[0]
    assert metric == "recall@5"
    assert drop == pytest.approx(0.06)


def test_compare_to_baseline_no_failures_when_all_within_threshold():
    baseline = {"recall@5": 0.80}
    current = {"recall@5": 0.76}
    assert compare_to_baseline(current, baseline, threshold=0.05) == []


def test_compare_to_baseline_skips_metrics_only_in_current():
    baseline = {"recall@5": 0.80}
    current = {"recall@5": 0.75, "ndcg@5": 0.90}
    failures = compare_to_baseline(current, baseline)
    # ndcg@5 not in baseline => skipped
    assert len(failures) == 1
    assert failures[0][0] == "recall@5"


def test_latency_p50_and_p90():
    report = RetrievalEvalReport(
        dataset_path="d.yaml",
        collection="c",
        k=5,
        rows=[
            PerQueryRow(
                qid=f"Q{i}", metrics={}, latency_ms=float(i * 10), retrieved_top_ids=[]
            )
            for i in range(1, 11)  # 10, 20, 30, ..., 100
        ],
    )
    # p50 of [10,20,...,100] => median of 10 values => avg of 5th(50) and 6th(60) = 55
    assert report.latency_p50_ms == pytest.approx(55.0)
    # p90: index = round(0.9 * 9) = 8 => sorted[8] = 90
    assert report.latency_p90_ms == pytest.approx(90.0)


def test_empty_report_aggregate_returns_empty_dict():
    report = RetrievalEvalReport(dataset_path="d.yaml", collection="c", k=5)
    assert report.aggregate == {}
    assert report.latency_p50_ms == 0.0
    assert report.latency_p90_ms == 0.0
