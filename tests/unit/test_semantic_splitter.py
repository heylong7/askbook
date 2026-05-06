"""Unit tests for SemanticSplitter (embedding-based semantic chunking)."""

from __future__ import annotations

import pytest

from askbook.core.exceptions import ChunkingError
from askbook.core.models import Document


class ControlledEmbedder:
    """Embedder that returns predefined vectors for specific texts.

    Used to control which sentences appear similar or dissimilar
    in semantic splitter tests.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors: dict[str, list[float]] = vectors or {}
        self._dim = 4

    def add_vector(self, text: str, vector: list[float]) -> None:
        self._vectors[text] = vector
        self._dim = len(vector)

    @property
    def model_name(self) -> str:
        return "controlled"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        return [self._vectors.get(t, [0.0] * self._dim) for t in texts]


def _make_doc(content: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, source_path="/tmp/x.md", content=content)


def test_semantic_splitter_empty_document_raises() -> None:
    """Empty document content raises ChunkingError."""
    from askbook.splitters.semantic import SemanticSplitter

    with pytest.raises(ChunkingError, match="Empty document"):
        SemanticSplitter().split(_make_doc(""))


def test_semantic_splitter_single_sentence() -> None:
    """Single sentence produces one chunk with correct metadata."""
    from askbook.splitters.semantic import SemanticSplitter

    chunks = SemanticSplitter().split(_make_doc("Hello world."))
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"
    assert chunks[0].content == "Hello world."
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["splitter"] == "semantic"
    assert chunks[0].metadata["source_path"] == "/tmp/x.md"


def test_semantic_splitter_multi_paragraph() -> None:
    """Multiple sentences with embedder produce chunks."""
    from askbook.splitters.semantic import SemanticSplitter

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a completely different topic about science. "
        "Quantum mechanics describes the behavior of particles. "
        "Another unrelated sentence about cooking recipes."
    )
    chunks = SemanticSplitter(embedder=ControlledEmbedder()).split(_make_doc(text))
    assert len(chunks) >= 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_semantic_splitter_merges_similar_sentences() -> None:
    """Similar sentences (identical embeddings) are merged into one chunk."""
    from askbook.splitters.semantic import SemanticSplitter

    embedder = ControlledEmbedder()
    s1 = "First sentence about AI."
    s2 = "Second sentence about AI."
    s3 = "Third sentence about AI."
    vec = [1.0, 0.0, 0.0, 0.0]
    embedder.add_vector(s1, vec)
    embedder.add_vector(s2, vec)
    embedder.add_vector(s3, vec)

    text = f"{s1} {s2} {s3}"
    chunks = SemanticSplitter(embedder=embedder, similarity_threshold=0.6).split(
        _make_doc(text)
    )

    # All similar -> 1 chunk (plus force-split only if oversized)
    assert len(chunks) == 1
    assert s1 in chunks[0].content
    assert s2 in chunks[0].content
    assert s3 in chunks[0].content


def test_semantic_splitter_splits_at_dissimilar_boundary() -> None:
    """Dissimilar sentences are split into separate chunks."""
    from askbook.splitters.semantic import SemanticSplitter

    embedder = ControlledEmbedder()
    s1 = "Topic A first part."
    s2 = "Topic A second part."
    s3 = "Topic B first part."
    s4 = "Topic B second part."
    # A-vectors are similar to each other, B-vectors are similar to each other,
    # but A and B vectors are orthogonal (dissimilar)
    vec_a = [1.0, 1.0, 0.0, 0.0]
    vec_b = [0.0, 0.0, 1.0, 1.0]
    embedder.add_vector(s1, vec_a)
    embedder.add_vector(s2, vec_a)
    embedder.add_vector(s3, vec_b)
    embedder.add_vector(s4, vec_b)

    text = f"{s1} {s2} {s3} {s4}"
    chunks = SemanticSplitter(embedder=embedder, similarity_threshold=0.6).split(
        _make_doc(text)
    )

    # Two topic groups -> at least 2 chunks
    assert len(chunks) >= 2
    # Topic A sentences should be in one chunk, Topic B in another
    chunk_texts = [c.content for c in chunks]
    topic_a_chunks = [c for c in chunk_texts if "Topic A" in c]
    topic_b_chunks = [c for c in chunk_texts if "Topic B" in c]
    assert len(topic_a_chunks) >= 1
    assert len(topic_b_chunks) >= 1


def test_semantic_splitter_without_embedder_falls_back() -> None:
    """Without embedder, all sentences go into a single chunk."""
    from askbook.splitters.semantic import SemanticSplitter

    text = "First sentence. Second sentence. Third sentence."
    chunks = SemanticSplitter(embedder=None).split(_make_doc(text))
    assert len(chunks) == 1
    assert "First sentence." in chunks[0].content
    assert "Second sentence." in chunks[0].content
    assert "Third sentence." in chunks[0].content


def test_semantic_splitter_cjk_text() -> None:
    """CJK text with Chinese punctuation is split correctly."""
    from askbook.splitters.semantic import SemanticSplitter

    text = (
        "人工智能是计算机科学的分支。"
        "机器学习是人工智能的核心技术。"
        "深度学习使用多层神经网络。"
    )
    chunks = SemanticSplitter().split(_make_doc(text))
    assert len(chunks) >= 1
    assert all(c.doc_id == "d1" for c in chunks)
    # Verify CJK content is preserved
    assert "人工智能" in chunks[0].content or "人工智能" in "".join(
        c.content for c in chunks
    )


def test_semantic_splitter_oversized_chunk_force_split() -> None:
    """When a chunk exceeds max_chunk_size, it is force-split by character count."""
    from askbook.splitters.semantic import SemanticSplitter

    # Create text that exceeds the max_chunk_size
    long_sentence = "word " * 300  # ~1500 chars
    chunks = SemanticSplitter(embedder=None, max_chunk_size=500).split(
        _make_doc(long_sentence)
    )

    # Should produce multiple chunks due to force split
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 500


def test_semantic_splitter_oversized_with_embedder() -> None:
    """Force split when embedder is present and a single chunk is oversized."""
    from askbook.splitters.semantic import SemanticSplitter

    embedder = ControlledEmbedder()
    s1 = "word " * 300  # ~1500 chars
    s2 = "Another short sentence."
    vec = [1.0, 0.0, 0.0, 0.0]
    embedder.add_vector(s1, vec)
    embedder.add_vector(s2, vec)

    text = f"{s1} {s2}"
    chunks = SemanticSplitter(
        embedder=embedder, similarity_threshold=0.6, max_chunk_size=500
    ).split(_make_doc(text))

    # s1 exceeds max_chunk_size so it gets force-split
    assert len(chunks) >= 2
