"""SemanticSplitter: embedding-based semantic chunking at paragraph boundaries."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from askbook.core.exceptions import ChunkingError
from askbook.core.interfaces import EmbedderProtocol
from askbook.core.models import Chunk, Document


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using CJK-aware boundaries."""
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


class SemanticSplitter:
    """Split documents at semantic boundaries using embedding similarity."""

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        similarity_threshold: float = 0.6,
        max_chunk_size: int = 1200,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_chunk_size = max_chunk_size

    def split(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            raise ChunkingError(f"Empty document: {document.doc_id}")

        sentences = _split_sentences(document.content)
        if not sentences:
            raise ChunkingError(f"No splittable content: {document.doc_id}")

        boundaries = self._find_boundaries(sentences)

        chunks: list[Chunk] = []
        start = 0
        for end in boundaries:
            piece = " ".join(sentences[start:end])
            if len(piece) > self._max_chunk_size:
                sub_pieces = self._force_split(piece, self._max_chunk_size)
                for sp in sub_pieces:
                    chunks.append(self._make_chunk(document, len(chunks), sp))
            else:
                chunks.append(self._make_chunk(document, len(chunks), piece))
            start = end

        if not chunks:
            raise ChunkingError(f"Splitter produced 0 chunks: {document.doc_id}")
        return chunks

    def _find_boundaries(self, sentences: list[str]) -> list[int]:
        """Find split boundaries where semantic similarity drops."""
        if len(sentences) <= 1 or self._embedder is None:
            return [len(sentences)]

        embeddings = self._embedder.embed_batch(sentences)
        boundaries: list[int] = []
        current_len = 0

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            current_len += len(sentences[i - 1])
            if sim < self._threshold or current_len > self._max_chunk_size:
                boundaries.append(i)
                current_len = 0

        boundaries.append(len(sentences))
        return boundaries

    def _force_split(self, text: str, max_size: int) -> list[str]:
        """Fallback: split oversized text at character boundary."""
        pieces: list[str] = []
        for i in range(0, len(text), max_size):
            pieces.append(text[i : i + max_size])
        return pieces

    def _make_chunk(self, document: Document, idx: int, content: str) -> Chunk:
        chunk_id = hashlib.sha256(
            f"{document.doc_id}:sem:{idx}:{content[:80]}".encode()
        ).hexdigest()
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            content=content,
            metadata={
                **document.metadata,
                "source_path": document.source_path,
                "chunk_index": idx,
                "splitter": "semantic",
            },
        )


__all__ = ["SemanticSplitter"]
