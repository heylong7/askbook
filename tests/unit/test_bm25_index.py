from pathlib import Path


def test_bm25_add_and_search_returns_ranked_results(tmp_path: Path) -> None:
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    idx = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    idx.add(
        [
            ("c1", "the quick brown fox jumps over the lazy dog", {}),
            ("c2", "askbook uses BM25 and dense retrieval", {}),
            ("c3", "hello world", {}),
        ]
    )
    hits = idx.search("askbook dense", top_k=2)
    assert hits[0][0] == "c2"
    assert len(hits) <= 2


def test_bm25_remove_drops_chunks(tmp_path: Path) -> None:
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    idx = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    idx.add([("c1", "alpha beta", {}), ("c2", "alpha gamma", {})])
    idx.remove(["c1"])
    assert {cid for cid, _ in idx.search("alpha", top_k=5)} == {"c2"}


def test_bm25_save_then_load_preserves_state(tmp_path: Path) -> None:
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    path = tmp_path / "bm25.pkl"
    idx = BM25PersistentIndex(path=path)
    idx.add([("c1", "askbook rag mcp", {}), ("c2", "unrelated content", {})])
    idx.save()

    reopened = BM25PersistentIndex(path=path)
    top1 = [h[0] for h in reopened.search("askbook", top_k=1)]
    assert top1 == ["c1"]


def test_bm25_search_empty_index_returns_empty_list(tmp_path: Path) -> None:
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    idx = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    assert idx.search("anything", top_k=5) == []


def test_search_as_results_returns_bm25_method_and_snippet_cap() -> None:
    import tempfile

    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25PersistentIndex(path=Path(tmpdir) / "idx.pkl")
        idx.add([("cid1", "hello world " * 20, {})])
        results = idx.search_as_results(
            "hello", top_k=5, snippet_lookup=lambda _: "X" * 500
        )
        assert len(results) == 1
        assert results[0].retrieval_method == "bm25"
        assert len(results[0].snippet) <= 200
