# ruff: noqa: N999
"""Page 5 — Evaluation results + Harness health metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from askbook.config.settings import load_settings
from askbook.dashboard.loader import load_events


def _save_feedback(question: str, fb_type: str) -> None:
    """Append feedback entry to eval_runs/feedback.jsonl."""
    feedback_path = Path.cwd() / "eval_runs" / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "question": question,
        "feedback": fb_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


st.title("评估与质量")

settings = load_settings()
trace_dir = Path(settings.observability.trace_dir)

# --- Harness metrics section ---
st.header("Harness 健康指标")

events = load_events(trace_dir, days=7)
mcp_nodes = {
    "search",
    "ask",
    "list_collections",
    "get_document_summary",
    "trace_lookup",
    "collection_stats",
}
mcp_events = [
    e for e in events if e.event_type == "span_end" and e.node_name in mcp_nodes
]
mcp_success = sum(1 for e in mcp_events if e.tags.get("status") == "success")
mcp_total = len(mcp_events)
total_retries = sum(
    1 for e in events if e.event_type == "span_end" and e.tags.get("retry") == "true"
)
task_count = sum(
    1
    for e in events
    if e.event_type == "span_end"
    and e.node_name == "ask"
    and e.tags.get("status") == "success"
)

pass_at_1_val = 0.0
golden_total = 0
eval_dir = Path.cwd() / "eval_runs"
latest_eval = sorted(eval_dir.glob("eval_*.json"), reverse=True)
if latest_eval:
    report = json.loads(latest_eval[0].read_text(encoding="utf-8"))
    golden_total = len(report.get("rows", []))
    golden_pass = sum(1 for r in report.get("rows", []) if r.get("hit", False))
    pass_at_1_val = golden_pass / golden_total if golden_total else 0.0

total_cost = sum(float(e.tags.get("estimated_cost_cny", 0)) for e in mcp_events)

col1, col2, col3, col4 = st.columns(4)
with col1:
    cr = mcp_success / mcp_total if mcp_total else 1.0
    st.metric(
        "Completion Rate",
        f"{cr:.1%}",
        delta=None if cr >= 0.95 else f"{cr - 0.95:.1%}",
        delta_color="off" if cr >= 0.95 else "inverse",
    )
with col2:
    rpt = total_retries / task_count if task_count else 0.0
    st.metric(
        "Retries/Task",
        f"{rpt:.2f}",
        delta=None if rpt <= 1.2 else f"{rpt - 1.2:.2f}",
        delta_color="off" if rpt <= 1.2 else "inverse",
    )
with col3:
    st.metric(
        "Pass@1",
        f"{pass_at_1_val:.1%}",
        delta=None if pass_at_1_val >= 0.85 else f"{pass_at_1_val - 0.85:.1%}",
        delta_color="off" if pass_at_1_val >= 0.85 else "inverse",
    )
with col4:
    cpt = total_cost / task_count if task_count else 0.0
    st.metric(
        "Cost/Task",
        f"¥{cpt:.4f}",
        delta=None if cpt <= 0.05 else f"¥{cpt - 0.05:.4f}",
        delta_color="off" if cpt <= 0.05 else "inverse",
    )

with st.expander("目标阈值"):
    st.markdown("""
    | 指标 | 目标 |
    |------|------|
    | Completion Rate | >= 95% |
    | Retries/Task | <= 1.2 |
    | Pass@1 | >= 85% |
    | Cost/Task | <= ¥0.05 |
    """)

# --- Retrieval metrics section ---
st.header("检索评估")
if latest_eval:
    agg = report.get("aggregate", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recall@5", f"{agg.get('recall@5', 0):.3f}")
        st.metric("MRR", f"{agg.get('mrr', 0):.3f}")
    with col2:
        st.metric("Hit Rate", f"{agg.get('hit_rate', 0):.3f}")
        st.metric("NDCG", f"{agg.get('ndcg@5', 0):.3f}")
    with col3:
        st.metric("Faithfulness", "—")
        st.metric("Answer Relevancy", "—")
else:
    st.info("评估数据需运行 `askbook eval` 命令后生成")

# --- User feedback section ---
st.header("用户反馈")
feedback_path = Path.cwd() / "eval_runs" / "feedback.jsonl"
if feedback_path.exists():
    with open(feedback_path, encoding="utf-8") as f:
        feedback_lines = [json.loads(line) for line in f if line.strip()]
    up_count = sum(1 for fb in feedback_lines if fb.get("feedback") == "up")
    down_count = sum(1 for fb in feedback_lines if fb.get("feedback") == "down")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👍", up_count)
    with col2:
        st.metric("👎", down_count)
else:
    st.info("暂无反馈数据")

# --- Feedback submission section ---
st.header("提交反馈")
feedback_q = st.text_input("问题（可选）", key="feedback_question")
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("有用", key="feedback_up"):
        _save_feedback(feedback_q, "up")
        st.success("已记录")
with col2:
    if st.button("无用", key="feedback_down"):
        _save_feedback(feedback_q, "down")
        st.success("已记录")
