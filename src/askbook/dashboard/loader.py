"""Dashboard data loader: reads JSONL trace files and computes metrics."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from askbook.observability.schema import TraceEvent

logger = logging.getLogger(__name__)


def load_events(trace_dir: Path, days: int = 7) -> list[TraceEvent]:
    """Load TraceEvent records from JSONL files in the last *days* days.

    Files are expected to be named ``YYYY-MM-DD.jsonl``.  Any line that cannot
    be parsed is skipped with a ``logger.warning``.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=days - 1)).date()
    results: list[TraceEvent] = []

    for jsonl_file in sorted(trace_dir.glob("*.jsonl")):
        stem = jsonl_file.stem
        try:
            file_date = datetime.strptime(stem, "%Y-%m-%d").date()
        except ValueError:
            # Not a date-named file — skip silently.
            continue

        if file_date < cutoff:
            continue

        for lineno, raw in enumerate(
            jsonl_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line:
                continue
            try:
                results.append(TraceEvent.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed trace line %s:%d — %s",
                    jsonl_file.name,
                    lineno,
                    exc,
                )

    return results


def aggregate_query_metrics(events: Sequence[TraceEvent]) -> dict[str, float]:
    """Compute count, p50_ms, p90_ms, total_tokens from span_end events.

    Returns zeros when *events* is empty.
    """
    span_ends = [ev for ev in events if ev.event_type == "span_end"]

    if not span_ends:
        return {"count": 0, "p50_ms": 0.0, "p90_ms": 0.0, "total_tokens": 0.0}

    durations = sorted(ev.duration_ms for ev in span_ends if ev.duration_ms is not None)

    if not durations:
        return {
            "count": float(len(span_ends)),
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "total_tokens": 0.0,
        }

    total_tokens = float(
        sum(
            int(ev.tags.get("tokens", 0))
            for ev in span_ends
            if ev.tags.get("tokens") is not None
        )
    )

    if len(durations) == 1:
        p50 = durations[0]
        p90 = durations[0]
    else:
        qs = statistics.quantiles(durations, n=100, method="inclusive")
        p50 = qs[49]
        p90 = qs[89]

    return {
        "count": float(len(span_ends)),
        "p50_ms": p50,
        "p90_ms": p90,
        "total_tokens": total_tokens,
    }


def aggregate_ingestion_runs(events: Sequence[TraceEvent]) -> list[dict[str, Any]]:
    """Group span_end events by trace_id; return one summary dict per ingestion run.

    Each summary contains:
    - ``trace_id``
    - ``total_duration_ms`` (sum of span durations)
    - ``node_count`` (number of span_end events)
    - ``source_path`` (from ``load`` span tags if present)
    """
    by_trace: dict[str, list[TraceEvent]] = defaultdict(list)
    for ev in events:
        if ev.event_type == "span_end":
            by_trace[ev.trace_id].append(ev)

    summaries: list[dict[str, Any]] = []
    for trace_id, spans in by_trace.items():
        total_duration = sum(s.duration_ms for s in spans if s.duration_ms is not None)
        source_path: str | None = None
        for span in spans:
            candidate = span.tags.get("source_path")
            if candidate is not None:
                source_path = str(candidate)
                break

        summaries.append(
            {
                "trace_id": trace_id,
                "total_duration_ms": total_duration,
                "node_count": len(spans),
                "source_path": source_path,
            }
        )

    return summaries


def trace_dir_mtime_signature(trace_dir: Path) -> tuple[float, ...]:
    """Return sorted mtimes of JSONL files in *trace_dir*.

    Used as a ``@st.cache_data`` hash key so Streamlit re-runs when files
    change.
    """
    mtimes = sorted(p.stat().st_mtime for p in trace_dir.glob("*.jsonl") if p.is_file())
    return tuple(mtimes)
