"""QA dataset schema and loader — DEV_SPEC Ch 25."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when seed_manual.yaml fails schema or invariant checks."""


class QAItemMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    reviewed_by: str = ""
    reviewed_at: str = ""
    collection: str = "demo"
    notes: str = ""


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qid: str
    question: str = Field(min_length=1)
    ground_truth: str = Field(default="")
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    provenance: Literal["manual", "llm_generated", "user_feedback"] = "manual"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_relevance(self) -> QAItem:
        if not self.relevant_chunk_ids and "empty_source" not in self.tags:
            msg = (
                f"qid={self.qid}: relevant_chunk_ids may be empty"
                " only when tags include 'empty_source'"
            )
            raise DatasetValidationError(msg)
        return self


class QADataset(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    meta: QAItemMeta = Field(alias="_meta")
    items: list[QAItem]

    @model_validator(mode="after")
    def _unique_qids(self) -> QADataset:
        seen: set[str] = set()
        for it in self.items:
            if it.qid in seen:
                raise DatasetValidationError(f"duplicate qid: {it.qid}")
            seen.add(it.qid)
            if it.provenance != "manual":
                logger.warning(
                    "qid=%s has provenance=%s (non-manual)", it.qid, it.provenance
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> QADataset:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise DatasetValidationError(f"invalid yaml at {path}: {exc}") from exc
        return cls.model_validate(raw)


__all__ = ["QAItem", "QAItemMeta", "QADataset", "DatasetValidationError"]
