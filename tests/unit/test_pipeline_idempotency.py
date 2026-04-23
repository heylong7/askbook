from pathlib import Path


def test_pipeline_run_is_idempotent_on_unchanged_source(tmp_path: Path) -> None:
    from askbook.embeddings.stub import StubEmbedder
    from askbook.ingestion.pipeline import IngestionPipeline
    from askbook.vectorstores.bm25_index import BM25PersistentIndex
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# A\n\nalpha beta gamma\n" * 10, encoding="utf-8")
    (docs_dir / "b.txt").write_text("body content " * 30, encoding="utf-8")

    data_dir = tmp_path / "data"
    store = ChromaVectorStore(path=str(data_dir / "chroma"))
    bm25 = BM25PersistentIndex(path=data_dir / "bm25" / "demo.pkl")
    pipeline = IngestionPipeline(
        embedder=StubEmbedder(dimension=8),
        store=store,
        bm25_index=bm25,
        chunk_size=80,
        chunk_overlap=20,
    )

    first = pipeline.run(source=docs_dir, collection="demo")
    assert first.chunks_added > 0
    assert first.chunks_reused == 0
    assert first.chunks_deleted == 0

    second = pipeline.run(source=docs_dir, collection="demo")
    assert second.chunks_added == 0
    assert second.chunks_reused == first.chunks_added
    assert second.chunks_deleted == 0


def test_pipeline_reacts_to_file_update(tmp_path: Path) -> None:
    import time

    from askbook.embeddings.stub import StubEmbedder
    from askbook.ingestion.pipeline import IngestionPipeline
    from askbook.vectorstores.bm25_index import BM25PersistentIndex
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "keep.md"
    target.write_text("# A\n\nfirst version\n", encoding="utf-8")

    data_dir = tmp_path / "data"
    pipeline = IngestionPipeline(
        embedder=StubEmbedder(dimension=8),
        store=ChromaVectorStore(path=str(data_dir / "chroma")),
        bm25_index=BM25PersistentIndex(path=data_dir / "bm25" / "demo.pkl"),
        chunk_size=80,
        chunk_overlap=20,
    )
    pipeline.run(source=docs, collection="demo")

    time.sleep(0.01)
    target.write_text(
        "# A\n\nSECOND version entirely rewritten body\n", encoding="utf-8"
    )

    result = pipeline.run(source=docs, collection="demo")
    assert result.chunks_added >= 1
    assert result.chunks_deleted >= 1
