from pathlib import Path


def _mk(tmp_path: Path):  # type: ignore[no-untyped-def]
    from askbook.embeddings.stub import StubEmbedder
    from askbook.ingestion.pipeline import IngestionPipeline
    from askbook.vectorstores.bm25_index import BM25PersistentIndex
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    data = tmp_path / "data"
    store = ChromaVectorStore(path=str(data / "chroma"))
    bm25 = BM25PersistentIndex(path=data / "bm25" / "demo.pkl")
    pipeline = IngestionPipeline(
        embedder=StubEmbedder(dimension=8),
        store=store,
        bm25_index=bm25,
        chunk_size=80,
        chunk_overlap=20,
    )
    return pipeline, store


def test_pipeline_dry_run_writes_nothing(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# X\n" + "alpha " * 200, encoding="utf-8")

    pipeline, store = _mk(tmp_path)
    result = pipeline.run(source=docs, collection="demo", dry_run=True)
    assert result.chunks_added == 0
    stats = store.get_collection_stats(store.make_collection_name("demo", "stub"))
    assert stats.chunk_count == 0


def test_pipeline_force_reindex_resets_existing(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n" + "alpha " * 100, encoding="utf-8")

    pipeline, store = _mk(tmp_path)
    pipeline.run(source=docs, collection="demo")

    result = pipeline.run(source=docs, collection="demo", force_reindex=True)
    assert result.chunks_added > 0
    assert result.chunks_reused == 0
