"""Streamlit entry point for the askbook dashboard."""

from __future__ import annotations

import streamlit as st

from askbook.config import load_settings

st.set_page_config(page_title="askbook", layout="wide")
cfg = load_settings()
st.session_state.setdefault("cfg", cfg)
st.sidebar.title("askbook Dashboard")
st.sidebar.markdown(
    "- 1 系统总览\n- 2 数据浏览\n- 3 Pipeline 监控\n- 4 Trace 查看器\n- 5 评估结果"
)
