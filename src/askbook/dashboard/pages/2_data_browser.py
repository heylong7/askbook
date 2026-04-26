# ruff: noqa: N999
"""Page 2 — Read-only data browser for Chroma collections."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from askbook.vectorstores.chroma_store import ChromaVectorStore

cfg = st.session_state.get("cfg")
if cfg is None:
    from askbook.config import load_settings

    cfg = load_settings()

try:
    store = ChromaVectorStore(path=str(Path(cfg.vectorstore.path).expanduser()))
except Exception as exc:  # noqa: BLE001
    st.error(f"无法初始化向量库: {exc}")
    st.stop()

st.title("数据浏览")

try:
    collections = store.list_collections()
except Exception as exc:  # noqa: BLE001
    st.warning(f"无法连接到向量库: {exc}")
    collections = []

if not collections:
    st.info("没有找到任何 Collection")
    st.stop()

names = [c.name for c in collections]
sel = st.selectbox("Collection", names)

if sel:
    try:
        docs = store.list_documents(sel, limit=200)
        if docs:
            st.dataframe(docs)
            chosen_doc = st.selectbox("选择文档", [d["doc_id"] for d in docs])
            if chosen_doc:
                chunks_list = store.get_document_chunks(str(chosen_doc), sel)
                for chunk in chunks_list[:20]:
                    st.text(chunk.content[:200])
        else:
            st.info("该 Collection 暂无文档数据")
    except Exception as exc:  # noqa: BLE001
        st.error(f"读取 Collection 时出错: {exc}")
