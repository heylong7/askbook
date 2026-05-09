"""Chroma persistent vector store implementing VectorStoreABC."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import chromadb

from askbook.core.interfaces import VectorStoreABC
from askbook.core.models import (
    Chunk,
    CollectionInfo,
    CollectionStats,
    RetrievalResult,
)


def _snip(text: str, n: int = 200) -> str:
    return text[:n]


class ChromaVectorStore(VectorStoreABC):
    def __init__(self, path: str) -> None:
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(p))

    def _get_collection(self, name: str) -> Any:
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], collection: str) -> int:
        if not chunks:
            return 0
        col = self._get_collection(collection)
        ids = [c.chunk_id for c in chunks]
        embeds = [c.embedding for c in chunks]
        docs = [c.content for c in chunks]
        metas = [{**c.metadata, "doc_id": c.doc_id} for c in chunks]
        if any(e is None for e in embeds):
            raise ValueError("All chunks must have embedding before upsert")
        col.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
        return len(chunks)

    def delete(self, doc_ids: list[str], collection: str) -> int:
        if not doc_ids:
            return 0
        col = self._get_collection(collection)
        total = 0
        for doc_id in doc_ids:
            matches = col.get(where={"doc_id": doc_id}, include=[])
            ids_to_delete = matches.get("ids", [])
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                total += len(ids_to_delete)
        return total

    def search(
        self,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        col = self._get_collection(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters
        res = col.query(**kwargs)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[RetrievalResult] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            score = 1.0 - float(dist)
            out.append(
                RetrievalResult(
                    chunk_id=cid,
                    score=score,
                    snippet=_snip(doc or ""),
                    metadata=dict(meta or {}),
                    retrieval_method="dense",
                )
            )
        return out

    def list_collections(self) -> list[CollectionInfo]:
        out: list[CollectionInfo] = []
        for c in self._client.list_collections():
            name = c.name
            parts = name.split("__")
            embed_model = parts[1] if len(parts) >= 2 else "unknown"
            out.append(
                CollectionInfo(name=name, size=c.count(), embed_model=embed_model)
            )
        return out

    def get_collection_stats(self, collection: str) -> CollectionStats:
        col = self._get_collection(collection)
        count = col.count()
        doc_ids: set[str] = set()
        if count > 0:
            data = col.get(include=["metadatas"])
            for m in data.get("metadatas", []) or []:
                if m and "doc_id" in m:
                    doc_ids.add(str(m["doc_id"]))
        return CollectionStats(
            collection=collection,
            chunk_count=count,
            doc_count=len(doc_ids),
            last_updated=_dt.datetime.now(_dt.UTC).isoformat(),
        )

    def get_document_chunks(self, doc_id: str, collection: str) -> list[Chunk]:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return []
        res = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        ids: list[str] = res.get("ids", []) or []
        docs: list[str] = res.get("documents", []) or []
        raw_metas = res.get("metadatas", []) or []
        metas: list[dict[str, Any]] = [
            dict(m) if m is not None else {} for m in raw_metas
        ]
        return [
            Chunk(
                chunk_id=str(cid),
                doc_id=doc_id,
                content=str(doc or ""),
                metadata=dict(meta or {}),
            )
            for cid, doc, meta in zip(ids, docs, metas, strict=False)
        ]

    def get_chunk_by_id(self, chunk_id: str, collection: str) -> Chunk | None:
        try:
            col = self._client.get_collection(collection)
        except Exception:
            return None
        res = col.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids: list[str] = res.get("ids", []) or []
        if not ids:
            return None
        docs = res.get("documents", []) or [""]
        metas_raw = (res.get("metadatas", []) or [{}])
        meta: dict[str, Any] = dict(metas_raw[0]) if metas_raw[0] else {}
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            content=str(docs[0] or ""),
            metadata=meta,
        )

    def list_documents(self, collection: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return one summary dict per unique doc_id in the collection.

        Each dict contains: doc_id (str), source_path (str), chunk_count (int).
        """
        col = self._get_collection(collection)
        result = col.get(include=["metadatas"], limit=limit)
        metadatas: list[dict[str, Any]] = [
            dict(m) if m is not None else {} for m in (result.get("metadatas") or [])
        ]
        docs: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source_path": str(meta.get("source_path", "")),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] = docs[doc_id]["chunk_count"] + 1
        return list(docs.values())

    def list_chunk_ids_by_doc(self, doc_id: str, collection: str) -> list[str]:
        col = self._get_collection(collection)
        res = col.get(where={"doc_id": doc_id}, include=[])
        return list(res.get("ids", []))

    def list_chunk_ids_by_source_path(
        self, source_path: str, collection: str
    ) -> list[str]:
        """Return all chunk IDs whose metadata.source_path matches *source_path*."""
        col = self._get_collection(collection)
        res = col.get(where={"source_path": source_path}, include=[])
        return list(res.get("ids", []))


__all__ = ["ChromaVectorStore"]
