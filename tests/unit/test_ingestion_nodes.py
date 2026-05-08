from pathlib import Path
from typing import Any


def _ctx(**kw: Any) -> dict[str, Any]:
    return dict(kw)


def test_document_loader_node_fills_documents(tmp_path: Path) -> None:
    from askbook.ingestion.nodes import DocumentLoaderNode
    from askbook.observability.null_trace import NullTraceWriter

    (tmp_path / "a.md").write_text("# A\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text body", encoding="utf-8")

    node = DocumentLoaderNode(trace_writer=NullTraceWriter())
    out = node.run(_ctx(source_path=str(tmp_path)))
    assert len(out["documents"]) == 2
    assert {d.source_path.split("/")[-1] for d in out["documents"]} | {
        d.source_path.split("\\")[-1] for d in out["documents"]
    }


def test_splitter_node_flattens_documents_into_chunks() -> None:
    from askbook.core.models import Document
    from askbook.ingestion.nodes import SplitterNode
    from askbook.observability.null_trace import NullTraceWriter

    node = SplitterNode(trace_writer=NullTraceWriter(), chunk_size=60, chunk_overlap=10)
    docs = [
        Document(doc_id=f"d{i}", source_path=f"/x{i}", content="a" * 150)
        for i in range(2)
    ]
    out = node.run(_ctx(documents=docs))
    assert len(out["chunks"]) >= 4
    assert {c.doc_id for c in out["chunks"]} == {"d0", "d1"}


def test_enrichment_node_is_identity_passthrough() -> None:
    from askbook.core.models import Chunk
    from askbook.ingestion.nodes import EnrichmentNode
    from askbook.observability.null_trace import NullTraceWriter

    chunks = [Chunk(chunk_id="c1", doc_id="d1", content="x")]
    out = EnrichmentNode(trace_writer=NullTraceWriter()).run(_ctx(chunks=chunks))
    assert out["chunks"] == chunks


def test_dedup_node_populates_new_and_stale() -> None:
    from askbook.core.models import Chunk
    from askbook.ingestion.nodes import DedupNode
    from askbook.observability.null_trace import NullTraceWriter

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="x"),
        Chunk(chunk_id="c2", doc_id="d1", content="y"),
    ]
    out = DedupNode(trace_writer=NullTraceWriter()).run(
        _ctx(chunks=chunks, existing_chunk_ids={"c1", "c3"})
    )
    assert [c.chunk_id for c in out["new_chunks"]] == ["c2"]
    assert out["stale_chunk_ids"] == ["c3"]


def test_embedding_node_fills_embedding_on_new_chunks() -> None:
    from askbook.core.models import Chunk
    from askbook.embeddings.stub import StubEmbedder
    from askbook.ingestion.nodes import EmbeddingNode
    from askbook.observability.null_trace import NullTraceWriter

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="hello"),
        Chunk(chunk_id="c2", doc_id="d1", content="world"),
    ]
    out = EmbeddingNode(
        embedder=StubEmbedder(dimension=8), trace_writer=NullTraceWriter()
    ).run(_ctx(new_chunks=chunks))
    assert all(
        c.embedding is not None and len(c.embedding) == 8 for c in out["new_chunks"]
    )


def test_vector_store_write_node_upserts_and_deletes(tmp_path: Path) -> None:
    from askbook.core.models import Chunk
    from askbook.ingestion.nodes import VectorStoreWriteNode
    from askbook.observability.null_trace import NullTraceWriter
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name("kb", "stub")
    chunk = Chunk(chunk_id="c1", doc_id="d1", content="x", embedding=[0.1] * 4)
    store.upsert([chunk], collection=collection)

    new_chunk = Chunk(chunk_id="c2", doc_id="d1", content="y", embedding=[0.2] * 4)
    node = VectorStoreWriteNode(store=store, trace_writer=NullTraceWriter())
    out = node.run(
        _ctx(
            collection=collection,
            new_chunks=[new_chunk],
            stale_chunk_ids=["c1"],
        )
    )
    assert out is not None
    stats = store.get_collection_stats(collection)
    assert stats.chunk_count == 1


def test_bm25_update_node_adds_and_removes(tmp_path: Path) -> None:
    from askbook.core.models import Chunk
    from askbook.ingestion.nodes import BM25IndexUpdateNode
    from askbook.observability.null_trace import NullTraceWriter
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    idx = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    idx.add([("c1", "old content", {})])
    node = BM25IndexUpdateNode(index=idx, trace_writer=NullTraceWriter())
    node.run(
        _ctx(
            new_chunks=[Chunk(chunk_id="c2", doc_id="d1", content="new content")],
            stale_chunk_ids=["c1"],
        )
    )
    hits = {cid for cid, _ in idx.search("content", top_k=5)}
    assert hits == {"c2"}
