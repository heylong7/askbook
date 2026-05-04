"""RetrievalEvalReport: per-qid breakdown + aggregate means + baseline comparison."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PerQueryRow:
    qid: str
    metrics: dict[str, float]
    latency_ms: float
    retrieved_top_ids: list[str]


@dataclass
class RetrievalEvalReport:
    dataset_path: str
    collection: str
    k: int
    rows: list[PerQueryRow] = field(default_factory=list)

    @property
    def aggregate(self) -> dict[str, float]:
        if not self.rows:
            return {}
        names = list(self.rows[0].metrics.keys())
        return {n: statistics.fmean(r.metrics[n] for r in self.rows) for n in names}

    @property
    def latency_p50_ms(self) -> float:
        if not self.rows:
            return 0.0
        return statistics.median(r.latency_ms for r in self.rows)

    @property
    def latency_p90_ms(self) -> float:
        if not self.rows:
            return 0.0
        sorted_l = sorted(r.latency_ms for r in self.rows)
        idx = max(0, int(round(0.9 * (len(sorted_l) - 1))))
        return sorted_l[idx]

    def format_table(self) -> str:
        agg = self.aggregate
        lines = [f"Eval @ k={self.k} on {self.collection}", "-" * 48]
        for name, val in agg.items():
            lines.append(f"  {name:<12} {val:.4f}")
        lines.append(f"  latency_p50  {self.latency_p50_ms:.1f} ms")
        lines.append(f"  latency_p90  {self.latency_p90_ms:.1f} ms")
        return "\n".join(lines)

    def to_json(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "collection": self.collection,
            "k": self.k,
            "aggregate": self.aggregate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "rows": [asdict(r) for r in self.rows],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float = 0.05,
) -> list[tuple[str, float, float, float]]:
    """Return list of (metric, baseline, current, drop) where drop > threshold.

    Drop is absolute (baseline - current). Metrics present in current but not
    baseline are skipped (new metric -- should be added via --update-baseline).
    """
    failures: list[tuple[str, float, float, float]] = []
    for metric, base in baseline.items():
        cur = current.get(metric)
        if cur is None:
            continue
        drop = base - cur
        if drop > threshold:
            failures.append((metric, base, cur, drop))
    return failures


__all__ = ["PerQueryRow", "RetrievalEvalReport", "compare_to_baseline"]
