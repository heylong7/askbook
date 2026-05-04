"""Tests for RetrievalEvalRunner (integration-level with fake pipeline)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from askbook.core.models import RetrievalResult
from askbook.evaluation.datasets import QADataset, QAItem, QAItemMeta
from askbook.evaluation.runner import EvalRunnerError, RetrievalEvalRunner


@dataclass
class FakePipeline:
    """Stub that satisfies the _RetrieveOnlyPipeline protocol."""

    results_map: dict[str, list[RetrievalResult]] = field(default_factory=dict)
    delay: float = 0.0
    fail_questions: set[str] = field(default_factory=set)

    def run_retrieve_only(
        self, *, query: str, collection: str
    ) -> list[RetrievalResult]:
        if query in self.fail_questions:
            raise ValueError(f"pipeline error for {query}")
        time.sleep(self.delay)
        return self.results_map.get(query, [])


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.95,
        snippet="snippet",
        retrieval_method="bm25",
    )


def test_runner_computes_per_qid_and_aggregate(tmp_path: object) -> None:
    """Q1 hits at rank 1 (mrr=1.0), Q2 hits at rank 2 (mrr=0.5)."""
    q1 = QAItem(qid="Q1", question="What is X?", relevant_chunk_ids=["chunk_a"])
    q2 = QAItem(qid="Q2", question="What is Y?", relevant_chunk_ids=["chunk_b"])

    ds = QADataset(meta=QAItemMeta(version="1.0"), items=[q1, q2])

    pipeline = FakePipeline(
        results_map={
            "What is X?": [_result("chunk_a")],
            "What is Y?": [_result("chunk_x"), _result("chunk_b")],
        },
    )
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)
    report = runner.run(dataset=ds, collection="demo")

    assert len(report.rows) == 2
    assert report.aggregate["mrr"] == pytest.approx(0.75)


def test_runner_handles_empty_source_qid_zero_metrics(tmp_path: object) -> None:
    """Empty relevant_chunk_ids yields zero for all metrics, no crash."""
    q = QAItem(
        qid="Q000",
        question="Empty query",
        relevant_chunk_ids=[],
        tags=["empty_source"],
    )
    ds = QADataset(meta=QAItemMeta(version="1.0"), items=[q])
    pipeline = FakePipeline()
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)
    report = runner.run(dataset=ds, collection="demo")

    assert len(report.rows) == 1
    assert report.rows[0].qid == "Q000"
    for metric, val in report.rows[0].metrics.items():
        assert val == 0.0, f"{metric} should be 0.0, got {val}"


def test_runner_records_latency_ms(tmp_path: object) -> None:
    """Latency measurement captures pipeline duration."""
    q = QAItem(qid="Q1", question="Hi?", relevant_chunk_ids=["chunk_a"])
    ds = QADataset(meta=QAItemMeta(version="1.0"), items=[q])
    pipeline = FakePipeline(
        results_map={"Hi?": [_result("chunk_a")]},
        delay=0.005,
    )
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)
    report = runner.run(dataset=ds, collection="demo")

    assert report.rows[0].latency_ms >= 5.0


def test_runner_propagates_pipeline_exception_with_qid_context(
    tmp_path: object,
) -> None:
    """Pipeline exception is wrapped in EvalRunnerError with qid context."""
    q = QAItem(qid="Q2", question="Raise me", relevant_chunk_ids=["chunk_a"])
    ds = QADataset(meta=QAItemMeta(version="1.0"), items=[q])
    pipeline = FakePipeline(fail_questions={"Raise me"})
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)

    with pytest.raises(EvalRunnerError) as exc_info:
        runner.run(dataset=ds, collection="demo")

    assert "qid=Q2" in str(exc_info.value)
    assert "pipeline error" in str(exc_info.value)
