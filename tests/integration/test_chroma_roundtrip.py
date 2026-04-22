from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askbook.core.models import Chunk


def _make_chunks(n: int, doc_id: str = "d1") -> list["Chunk"]:
    from askbook.core.models import Chunk

    return [
        Chunk(
            chunk_id=f"{doc_id}-c{i}",
            doc_id=doc_id,
            content=f"chunk-{i}-text",
            embedding=[float(i), 0.0, 0.0, 0.0],
            metadata={"chunk_index": i},
        )
        for i in range(n)
    ]


def test_chroma_upsert_and_search(tmp_path: Path) -> None:
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name("kb_demo", "stub")

    chunks = _make_chunks(5)
    written = store.upsert(chunks, collection=collection)
    assert written == 5

    hits = store.search(
        query_embedding=[0.0, 0.0, 0.0, 0.0], collection=collection, top_k=3
    )
    assert len(hits) == 3
    for h in hits:
        assert h.retrieval_method == "dense"
        assert len(h.snippet) <= 200


def test_chroma_delete_by_doc(tmp_path: Path) -> None:
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name("kb", "stub")
    store.upsert(_make_chunks(3, doc_id="dA"), collection=collection)
    store.upsert(_make_chunks(2, doc_id="dB"), collection=collection)

    removed = store.delete(["dA"], collection=collection)
    assert removed == 3

    stats = store.get_collection_stats(collection)
    assert stats.chunk_count == 2
    assert stats.doc_count == 1


def test_chroma_list_chunk_ids_by_doc(tmp_path: Path) -> None:
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name("kb", "stub")
    store.upsert(_make_chunks(4, doc_id="dX"), collection=collection)

    ids = store.list_chunk_ids_by_doc(doc_id="dX", collection=collection)
    assert set(ids) == {"dX-c0", "dX-c1", "dX-c2", "dX-c3"}
    assert store.list_chunk_ids_by_doc("nope", collection=collection) == []


def test_chroma_list_collections(tmp_path: Path) -> None:
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    c1 = store.make_collection_name("alpha", "stub")
    c2 = store.make_collection_name("beta", "stub")
    store.upsert(_make_chunks(1, doc_id="x"), collection=c1)
    store.upsert(_make_chunks(1, doc_id="y"), collection=c2)

    names = {c.name for c in store.list_collections()}
    assert {c1, c2}.issubset(names)
