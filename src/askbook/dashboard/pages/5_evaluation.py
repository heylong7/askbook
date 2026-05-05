# ruff: noqa: N999
"""Page 5 — Evaluation results: metric cards and eval run summary."""

from __future__ import annotations

import streamlit as st

st.title("评估结果")

st.info("评估数据需运行 `askbook eval` 命令后生成")

# -- Metric cards in 3-column layout --
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Recall@5", value="—")
    st.metric("MRR", value="—")

with col2:
    st.metric("Hit Rate", value="—")
    st.metric("NDCG", value="—")

with col3:
    st.metric("Faithfulness", value="—")
    st.metric("Answer Relevancy", value="—")
