# ruff: noqa: N999
"""Page 4 — Trace Viewer: browse, filter, and inspect trace events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from askbook.dashboard.loader import load_events, trace_dir_mtime_signature
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

st.title("Trace 查看器")

if not events:
    st.info("暂无 Trace 事件数据")
    st.stop()

# -- Filter controls --
col1, col2, col3 = st.columns(3)
with col1:
    days = st.selectbox("天数", options=[1, 3, 7, 30], index=2)
with col2:
    all_types = sorted({e.event_type for e in events})
    selected_types = st.multiselect("事件类型", options=all_types, default=all_types)
with col3:
    node_filter = st.text_input("节点名称筛选", value="")

# Re-load if days changed
if days != 7:
    events = load_events(trace_dir, days=days)

# Apply filters
filtered = events
if selected_types:
    filtered = [e for e in filtered if e.event_type in selected_types]
if node_filter.strip():
    filtered = [
        e for e in filtered if node_filter.strip().lower() in e.node_name.lower()
    ]

st.caption(f"显示 {len(filtered)} / {len(events)} 条事件")

if not filtered:
    st.info("无匹配的 Trace 事件")
    st.stop()

# -- Dataframe --
df = pd.DataFrame(
    [
        {
            "trace_id": e.trace_id,
            "span_id": e.span_id,
            "node_name": e.node_name,
            "event_type": e.event_type,
            "duration_ms": e.duration_ms,
            "timestamp_utc": e.timestamp_utc,
        }
        for e in filtered
    ]
)

selected_rows = st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun",
)

# -- Selected row detail --
if selected_rows is not None and len(selected_rows.selection.rows) > 0:
    idx = selected_rows.selection.rows[0]
    event = filtered[idx]
    st.subheader(f"详情: {event.trace_id[:16]}... / {event.node_name}")
    st.json(event.tags)
