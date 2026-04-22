import pytest


@pytest.mark.requires_bge_m3
def test_bge_m3_dimension_and_batch() -> None:
    from askbook.embeddings.bge_m3 import BGEM3Embedder

    e = BGEM3Embedder()
    v = e.embed_passage("hello world")
    assert len(v) == 1024
    batch = e.embed_batch(["a", "b"], is_query=False)
    assert len(batch) == 2 and len(batch[0]) == 1024
