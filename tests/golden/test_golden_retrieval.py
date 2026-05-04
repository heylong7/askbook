"""Golden retrieval regression — DEV_SPEC Ch 26.

Runs the real seed_manual.yaml against a real-ish pipeline and asserts that
no metric drops more than SCORE_DROP_THRESHOLD (= 0.05) below the recorded baseline.

Marked @pytest.mark.golden so it can be selected/excluded via -m.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askbook.evaluation.datasets import QADataset
from askbook.evaluation.report import compare_to_baseline
from askbook.evaluation.runner import RetrievalEvalRunner

DATASET_PATH = Path("datasets/seed_manual.yaml")
BASELINE_PATH = Path("tests/golden/baselines/v0.1_scores.json")
SCORE_DROP_THRESHOLD = 0.05


@pytest.fixture(scope="module")
def baseline() -> dict[str, float]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline_and_collection():
    from askbook.evaluation.cli import _build_pipeline

    try:
        pipeline, full, trace = _build_pipeline(config_path=None, collection="demo")
    except Exception as exc:
        pytest.skip(f"cannot build real pipeline (Ollama/ChromaDB unavailable?): {exc}")
    yield pipeline, full
    trace.flush()


@pytest.mark.golden
def test_retrieval_no_regression(pipeline_and_collection, baseline):
    if not DATASET_PATH.exists():
        pytest.skip("seed_manual.yaml missing")
    pipeline, full = pipeline_and_collection
    ds = QADataset.from_yaml(DATASET_PATH)
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)
    report = runner.run(dataset=ds, collection=full)
    failures = compare_to_baseline(report.aggregate, baseline, SCORE_DROP_THRESHOLD)
    assert not failures, "regression: " + " ; ".join(
        f"{m}: {b:.3f} -> {c:.3f} (drop={d:.3f})" for m, b, c, d in failures
    )


@pytest.mark.golden
def test_empty_source_qid_returns_zero_metrics(pipeline_and_collection):
    """Q000 with relevant=[] must produce all-zero metrics (no false positives)."""
    pipeline, full = pipeline_and_collection
    ds = QADataset.from_yaml(DATASET_PATH)
    q000 = next(it for it in ds.items if it.qid == "Q000")
    runner = RetrievalEvalRunner(pipeline=pipeline, k=5)
    row = runner._run_one(q000, full)
    assert all(v == 0.0 for v in row.metrics.values()), row.metrics
