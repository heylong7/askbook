def test_stub_embedder_is_deterministic_and_runtime_checkable() -> None:
    from askbook.core.interfaces import EmbedderProtocol
    from askbook.embeddings.stub import StubEmbedder

    e = StubEmbedder(dimension=8)
    assert isinstance(e, EmbedderProtocol)
    assert e.dimension == 8
    assert e.model_name == "stub"

    v1 = e.embed_passage("hello")
    v2 = e.embed_passage("hello")
    assert v1 == v2
    assert len(v1) == 8
    assert all(isinstance(x, float) for x in v1)


def test_stub_embedder_query_and_passage_differ() -> None:
    from askbook.embeddings.stub import StubEmbedder

    e = StubEmbedder(dimension=8)
    assert e.embed_query("same text") != e.embed_passage("same text")


def test_stub_embedder_batch_matches_single() -> None:
    from askbook.embeddings.stub import StubEmbedder

    e = StubEmbedder(dimension=8)
    batch = e.embed_batch(["a", "b", "c"], is_query=False)
    singles = [e.embed_passage(t) for t in ["a", "b", "c"]]
    assert batch == singles


def test_bge_m3_class_exposes_query_and_passage_prefix_constants() -> None:
    from askbook.embeddings.bge_m3 import BGEM3Embedder

    assert BGEM3Embedder.PASSAGE_PREFIX == ""
    assert isinstance(BGEM3Embedder.QUERY_PREFIX, str)
