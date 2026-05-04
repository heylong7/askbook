"""Integration tests for ``askbook eval`` CLI command."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from askbook.cli import app
from askbook.core.models import RetrievalResult

# ── YAML fixtures ────────────────────────────────────────────────────────────────

VALID_2_ITEM_YAML = """\
_meta:
  version: "0.1"
  reviewed_by: ""
  reviewed_at: ""
  collection: demo
  notes: "eval cli test dataset"
items:
  - qid: Q001
    question: "What is askbook?"
    ground_truth: "A local RAG system."
    relevant_chunk_ids:
      - chunk_a
    relevant_doc_ids:
      - doc_1
    provenance: manual
    tags:
      - retrieval
  - qid: Q002
    question: "How does ingestion work?"
    ground_truth: "It parses and chunks documents."
    relevant_chunk_ids:
      - chunk_b
    relevant_doc_ids:
      - doc_2
    provenance: manual
    tags:
      - retrieval
"""


# ── Fake test doubles ────────────────────────────────────────────────────────────


@dataclass
class FakePipeline:
    """Stub that satisfies the _RetrieveOnlyPipeline protocol."""

    results_map: dict[str, list[RetrievalResult]] = field(default_factory=dict)

    def run_retrieve_only(
        self, *, query: str, collection: str
    ) -> list[RetrievalResult]:
        return self.results_map.get(query, [])


@dataclass
class FakeTrace:
    """Stub trace writer with a no-op flush."""

    def flush(self) -> None:
        pass


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.95,
        snippet="snippet",
        retrieval_method="bm25",
    )


@pytest.fixture
def fake_pipeline_factory() -> Any:
    """Fixture that returns a fake _build_pipeline compatible with the CLI."""

    def _factory(
        config_path: Path | None, collection: str
    ) -> tuple[FakePipeline, str, FakeTrace]:
        pipeline = FakePipeline(
            results_map={
                "What is askbook?": [_result("chunk_a")],
                "How does ingestion work?": [_result("chunk_b")],
            },
        )
        return pipeline, f"{collection}-full", FakeTrace()

    return _factory


# ── Tests ────────────────────────────────────────────────────────────────────────


def test_eval_cli_writes_report_json(
    tmp_path: Path, monkeypatch: Any, fake_pipeline_factory: Any
) -> None:
    """End-to-end: CLI invocation writes a JSON report with aggregate + rows."""
    monkeypatch.setattr("askbook.evaluation.cli._build_pipeline", fake_pipeline_factory)
    ds_path = tmp_path / "ds.yaml"
    ds_path.write_text(VALID_2_ITEM_YAML, encoding="utf-8")
    out = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--dataset",
            str(ds_path),
            "--collection",
            "demo",
            "--k",
            "5",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "aggregate" in payload
    assert "rows" in payload


def test_eval_cli_update_baseline_writes_baseline_file(
    tmp_path: Path, monkeypatch: Any, fake_pipeline_factory: Any
) -> None:
    """--update-baseline flag writes an aggregate baseline JSON file."""
    monkeypatch.setattr("askbook.evaluation.cli._build_pipeline", fake_pipeline_factory)
    ds_path = tmp_path / "ds.yaml"
    ds_path.write_text(VALID_2_ITEM_YAML, encoding="utf-8")
    out = tmp_path / "report.json"
    baseline = tmp_path / "baselines" / "demo.json"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--dataset",
            str(ds_path),
            "--collection",
            "demo",
            "--k",
            "5",
            "--output",
            str(out),
            "--update-baseline",
            str(baseline),
        ],
    )
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    assert baseline.exists()
    content = json.loads(baseline.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    # Should contain the aggregate metrics
    for key in ("hit_rate", "mrr", "recall@5", "ndcg@5"):
        assert key in content, f"missing metric {key} in baseline"


def test_eval_cli_missing_dataset_returns_nonzero(tmp_path: Path) -> None:
    """Missing dataset file produces non-zero exit code and a helpful message."""
    result = CliRunner().invoke(
        app, ["eval", "--dataset", str(tmp_path / "missing.yaml")]
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
