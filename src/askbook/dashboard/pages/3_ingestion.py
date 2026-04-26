# ruff: noqa: N999
"""Page 3 — Ingestion monitor with Plotly Gantt chart."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st

from askbook.dashboard.loader import (
    aggregate_ingestion_runs,
    load_events,
    trace_dir_mtime_signature,
)
from askbook.observability.schema import TraceEvent

cfg = st.session_state.get("cfg")
if cfg is None:
    from askbook.config import load_settings

    cfg = load_settings()

trace_dir = Path(cfg.observability.trace_dir).expanduser()


@st.cache_data
def _load(_sig: tuple[float, ...]) -> list[TraceEvent]:
    return load_events(trace_dir, days=7)


events = _load(trace_dir_mtime_signature(trace_dir))
runs = aggregate_ingestion_runs(events)

st.title("Ingestion 监控")

if not runs:
    st.info("暂无 Ingestion Trace 数据")
    st.stop()

st.dataframe(runs)

trace_ids = [str(r.get("trace_id", "")) for r in runs if r.get("trace_id")]
if trace_ids:
    selected = st.selectbox("选择 Trace ID", trace_ids)
    if selected:
        spans = [
            e for e in events if e.trace_id == selected and e.event_type == "span_end"
        ]
        if spans:
            gantt_data: list[dict[str, Any]] = []
            for e in spans:
                duration = e.duration_ms or 0.0
                end_time = e.timestamp_utc
                start_time = end_time - timedelta(milliseconds=duration)
                gantt_data.append(
                    {
                        "node": e.node_name,
                        "start": start_time,
                        "end": end_time,
                    }
                )
            fig = px.timeline(
                gantt_data,
                x_start="start",
                x_end="end",
                y="node",
                title=f"Trace: {selected[:16]}...",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("该 Trace 暂无 span_end 事件")
