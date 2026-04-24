"""Integration tests for ChromaVectorStore.get_document_chunks."""

from __future__ import annotations

from pathlib import Path

from askbook.core.models import Chunk
from askbook.embeddings.stub import StubEmbedder
from askbook.vectorstores.chroma_store import ChromaVectorStore

COLLECTION = "demo__stub__v1"


def _make_chunks_with_embeddings(
    embedder: StubEmbedder,
    doc_id: str,
    count: int,
    id_offset: int = 0,
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}-c{i + id_offset}",
            doc_id=doc_id,
            content=f"content of {doc_id} chunk {i + id_offset}",
            embedding=embedder.embed_passage(
                f"content of {doc_id} chunk {i + id_offset}"
            ),
            metadata={"chunk_index": i + id_offset},
        )
        for i in range(count)
    ]


def test_get_document_chunks_filters_by_doc_id(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    embedder = StubEmbedder()

    chunks_a = _make_chunks_with_embeddings(embedder, "doc_A", 2)
    chunks_b = _make_chunks_with_embeddings(embedder, "doc_B", 1)

    store.upsert(chunks_a + chunks_b, COLLECTION)

    result = store.get_document_chunks("doc_A", COLLECTION)

    assert len(result) == 2
    assert all(c.doc_id == "doc_A" for c in result)


def test_get_document_chunks_unknown_doc_returns_empty(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    embedder = StubEmbedder()

    chunks_a = _make_chunks_with_embeddings(embedder, "doc_A", 2)
    chunks_b = _make_chunks_with_embeddings(embedder, "doc_B", 1)

    store.upsert(chunks_a + chunks_b, COLLECTION)

    result = store.get_document_chunks("doc_missing", COLLECTION)

    assert result == []


def test_get_document_chunks_unknown_collection_returns_empty(tmp_path: Path) -> None:
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))

    result = store.get_document_chunks("any_doc", "nonexistent__collection__v1")

    assert result == []
