"""Tests for QA dataset schema and loader — DEV_SPEC Ch 25."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from askbook.evaluation.datasets import DatasetValidationError, QADataset


def _make_valid_item(
    qid: str = "Q000",
    question: str = "What is askbook?",
    tags: list[str] | None = None,
    relevant_chunk_ids: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    item: dict = {
        "qid": qid,
        "question": question,
        "ground_truth": "Some answer.",
        "relevant_chunk_ids": relevant_chunk_ids
        if relevant_chunk_ids is not None
        else [],
        "relevant_doc_ids": [],
        "provenance": "manual",
        "tags": tags if tags is not None else ["empty_source"],
    }
    if extra:
        item.update(extra)
    return item


def _make_valid_dataset(items: list[dict]) -> dict:
    return {
        "_meta": {
            "version": "0.1",
            "reviewed_by": "",
            "reviewed_at": "",
            "collection": "demo",
            "notes": "test dataset",
        },
        "items": items,
    }


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "dataset.yaml"
    p.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ── Test 1 ───────────────────────────────────────────────────────────────────


def test_load_valid_yaml_returns_items_count_matches(tmp_path: Path) -> None:
    data = _make_valid_dataset(
        [
            _make_valid_item("Q000", tags=["empty_source"]),
            _make_valid_item("Q001", tags=["empty_source", "retrieval"]),
        ]
    )
    path = _write_yaml(tmp_path, data)
    ds = QADataset.from_yaml(path)
    assert len(ds.items) == 2
    assert ds.items[0].qid == "Q000"


# ── Test 2 ───────────────────────────────────────────────────────────────────


def test_qid_must_be_unique_raise(tmp_path: Path) -> None:
    data = _make_valid_dataset(
        [
            _make_valid_item("Q000"),
            _make_valid_item("Q000"),  # duplicate
        ]
    )
    path = _write_yaml(tmp_path, data)
    with pytest.raises(DatasetValidationError, match="duplicate qid"):
        QADataset.from_yaml(path)


# ── Test 3 ───────────────────────────────────────────────────────────────────


def test_missing_question_raises(tmp_path: Path) -> None:
    item = {
        "qid": "Q000",
        # "question" deliberately omitted
        "ground_truth": "answer",
        "relevant_chunk_ids": [],
        "relevant_doc_ids": [],
        "provenance": "manual",
        "tags": ["empty_source"],
    }
    data = _make_valid_dataset([item])
    path = _write_yaml(tmp_path, data)
    with pytest.raises(DatasetValidationError):
        QADataset.from_yaml(path)


# ── Test 4 ───────────────────────────────────────────────────────────────────


def test_empty_relevant_chunk_ids_allowed_with_empty_source_tag(tmp_path: Path) -> None:
    # Should pass: empty chunk IDs + empty_source tag
    data_ok = _make_valid_dataset(
        [_make_valid_item("Q000", relevant_chunk_ids=[], tags=["empty_source"])]
    )
    path_ok = _write_yaml(tmp_path, data_ok)
    ds = QADataset.from_yaml(path_ok)
    assert ds.items[0].relevant_chunk_ids == []

    # Should fail: empty chunk IDs without empty_source tag
    path_bad = tmp_path / "bad.yaml"
    data_bad = _make_valid_dataset(
        [_make_valid_item("Q000", relevant_chunk_ids=[], tags=["retrieval"])]
    )
    path_bad.write_text(yaml.dump(data_bad, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        QADataset.from_yaml(path_bad)


# ── Test 5 ───────────────────────────────────────────────────────────────────


def test_extra_field_raises(tmp_path: Path) -> None:
    data = _make_valid_dataset(
        [_make_valid_item("Q000", extra={"unexpected_field": "oops"})]
    )
    path = _write_yaml(tmp_path, data)
    with pytest.raises(DatasetValidationError):
        QADataset.from_yaml(path)


# ── Test 6 ───────────────────────────────────────────────────────────────────


def test_from_yaml_with_real_seed_manual() -> None:
    import os

    root = Path(os.environ.get("ASKBOOK_PROJECT_ROOT", ""))
    path = root / "datasets" / "seed_manual.yaml"
    if not path.exists():
        pytest.skip("seed_manual.yaml not present in this environment")
    ds = QADataset.from_yaml(path)
    assert len(ds.items) >= 1
    assert ds.meta.collection  # non-empty
