"""Integration test: seed_manual.yaml loads correctly with 20 items."""

from __future__ import annotations

from pathlib import Path

from askbook.evaluation.datasets import QADataset


def test_seed_manual_yaml_loads_and_has_20_items() -> None:
    ds = QADataset.from_yaml(Path("datasets/seed_manual.yaml"))
    assert len(ds.items) == 20
    qids = [it.qid for it in ds.items]
    assert len(set(qids)) == 20
    assert any("empty_source" in it.tags for it in ds.items)
    # human-in-the-loop check
    assert ds.meta.reviewed_by, "seed_manual must be human-reviewed (Harness 30.viii)"
