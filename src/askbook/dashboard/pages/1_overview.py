# ruff: noqa: N999
"""Page 1 — System overview: query metrics and health status."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from askbook.dashboard.health import HealthCheck
from askbook.dashboard.loader import (
    aggregate_query_metrics,
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
    return load_events(trace_dir, days=1)


events = _load(trace_dir_mtime_signature(trace_dir))
metrics = aggregate_query_metrics(events)

st.title("系统总览")

col1, col2, col3 = st.columns(3)
col1.metric("今日 Query 数", int(metrics["count"]))
col2.metric("P50 延迟 (ms)", f"{metrics['p50_ms']:.0f}")
col3.metric("Token 总量", int(metrics["total_tokens"]))

st.subheader("健康状态")
# Pass None for LLM and store so HealthCheck skips live-provider probes
health = HealthCheck(llm=None, store=None, bm25_dir=None).collect()
if health:
    for name, status in health.items():
        icon = "✅" if status["status"] == "up" else "❌"
        st.write(f"{icon} **{name}** — {status['status']}: {status.get('detail', '')}")
else:
    st.info("无可用健康检查项（所有组件均已跳过）")
