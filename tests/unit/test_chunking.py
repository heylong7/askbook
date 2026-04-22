from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askbook.core.models import Document


def _make_doc(content: str, doc_id: str = "d1") -> Document:
    from askbook.core.models import Document

    return Document(doc_id=doc_id, source_path="/tmp/x.md", content=content)


def test_splitter_returns_chunks_with_stable_chunk_ids() -> None:
    from askbook.splitters.recursive import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "abcdefghij" * 20  # 200 chars
    chunks = splitter.split(_make_doc(text))

    assert len(chunks) >= 3
    assert all(c.doc_id == "d1" for c in chunks)
    chunk_ids = {c.chunk_id for c in chunks}
    assert len(chunk_ids) == len(chunks)


def test_splitter_empty_content_raises_chunking_error() -> None:
    import pytest

    from askbook.core.exceptions import ChunkingError
    from askbook.splitters.recursive import RecursiveTextSplitter

    with pytest.raises(ChunkingError):
        RecursiveTextSplitter(chunk_size=50, chunk_overlap=10).split(_make_doc(""))


def test_splitter_preserves_doc_metadata_into_chunks() -> None:
    from askbook.core.models import Document
    from askbook.splitters.recursive import RecursiveTextSplitter

    doc = Document(
        doc_id="d2",
        source_path="/tmp/b.md",
        content="a" * 150,
        metadata={"suffix": ".md", "page": 1},
    )
    chunks = RecursiveTextSplitter(chunk_size=60, chunk_overlap=10).split(doc)
    assert all(c.metadata["suffix"] == ".md" for c in chunks)
    assert all(c.metadata["source_path"] == "/tmp/b.md" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_splitter_handles_mixed_chinese_english() -> None:
    from askbook.splitters.recursive import RecursiveTextSplitter

    text = "Hello world. 你好，世界。" * 20
    chunks = RecursiveTextSplitter(chunk_size=80, chunk_overlap=20).split(
        _make_doc(text)
    )
    joined = "".join(c.content for c in chunks)
    assert "你好" in joined and "Hello" in joined
