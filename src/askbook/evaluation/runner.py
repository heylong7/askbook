"""RetrievalEvalRunner: drives a pipeline over a QADataset and produces a report."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from askbook.core.models import RetrievalResult
from askbook.evaluation.datasets import QADataset, QAItem
from askbook.evaluation.metrics import hit_rate, mrr, ndcg_at_k, recall_at_k
from askbook.evaluation.report import PerQueryRow, RetrievalEvalReport


class EvalRunnerError(RuntimeError):
    """Raised when the pipeline fails during evaluation run."""


class _RetrieveOnlyPipeline(Protocol):
    """Minimal protocol satisfied by QueryPipeline.run_retrieve_only."""

    def run_retrieve_only(
        self, *, query: str, collection: str
    ) -> list[RetrievalResult]: ...


@dataclass
class RetrievalEvalRunner:
    """Orchestrates retrieval evaluation over a seed dataset.

    Uses a pipeline (real or fake) that satisfies the _RetrieveOnlyPipeline
    protocol, runs each QAItem through it, and produces a report with per-query
    metrics and aggregate scores.
    """

    pipeline: _RetrieveOnlyPipeline
    k: int = 5

    def run(self, *, dataset: QADataset, collection: str) -> RetrievalEvalReport:
        """Run all items in *dataset* and return a completed report."""
        report = RetrievalEvalReport(
            dataset_path="<provided>", collection=collection, k=self.k
        )
        for item in dataset.items:
            row = self._run_one(item, collection)
            report.rows.append(row)
        return report

    def _run_one(self, item: QAItem, collection: str) -> PerQueryRow:
        relevant = set(item.relevant_chunk_ids)
        t0 = time.perf_counter()
        try:
            results = self.pipeline.run_retrieve_only(
                query=item.question, collection=collection
            )
        except Exception as exc:
            raise EvalRunnerError(f"pipeline failed on qid={item.qid}: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.chunk_id for r in results]
        metrics = {
            "hit_rate": hit_rate(retrieved_ids, relevant, self.k),
            "mrr": mrr(retrieved_ids, relevant, self.k),
            f"recall@{self.k}": recall_at_k(retrieved_ids, relevant, self.k),
            f"ndcg@{self.k}": ndcg_at_k(retrieved_ids, relevant, self.k),
        }
        return PerQueryRow(
            qid=item.qid,
            metrics=metrics,
            latency_ms=latency_ms,
            retrieved_top_ids=retrieved_ids[: self.k],
        )


__all__ = ["RetrievalEvalRunner", "EvalRunnerError"]
