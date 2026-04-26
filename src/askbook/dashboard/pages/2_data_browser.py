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

store = ChromaVectorStore(path=str(Path(cfg.vectorstore.path).expanduser()))

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
        col_obj = store._get_collection(sel)
        result = col_obj.get(include=["metadatas"], limit=200)
        metadatas: list[dict[str, object]] = result.get("metadatas") or []

        docs: dict[str, dict[str, object]] = {}
        for meta in metadatas:
            if meta and "doc_id" in meta:
                doc_id = str(meta["doc_id"])
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "source_path": str(meta.get("source_path", "")),
                        "chunk_count": 0,
                    }
                prev = docs[doc_id]["chunk_count"]
                docs[doc_id]["chunk_count"] = (prev if isinstance(prev, int) else 0) + 1

        if docs:
            st.dataframe(list(docs.values()))
            chosen_doc = st.selectbox("选择文档", list(docs.keys()))
            if chosen_doc:
                chunks_list = store.get_document_chunks(chosen_doc, sel)
                for chunk in chunks_list[:20]:
                    st.text(chunk.content[:200])
        else:
            st.info("该 Collection 暂无文档数据")
    except Exception as exc:  # noqa: BLE001
        st.error(f"读取 Collection 时出错: {exc}")
