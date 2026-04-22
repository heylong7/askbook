"""Recursive character splitter — pure Python, no langchain dependency."""

from __future__ import annotations

import hashlib

from askbook.core.exceptions import ChunkingError
from askbook.core.models import Chunk, Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_text(
    text: str, separators: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Recursively split text by separators, then merge small pieces into chunks."""
    if not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep) if sep else list(text)

    # Recursively split large parts
    good: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            good.append(part)
        else:
            good.extend(_split_text(part, rest, chunk_size, chunk_overlap))

    # Merge small pieces back together with overlap
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in good:
        piece_len = len(piece)
        joiner = sep if sep else ""
        join_len = len(joiner) if current else 0

        if current_len + join_len + piece_len > chunk_size and current:
            chunk = joiner.join(current)
            if chunk.strip():
                chunks.append(chunk)
            # Keep overlap
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + (len(joiner) if current else 0)
            current_len = sum(len(p) for p in current) + len(joiner) * max(
                0, len(current) - 1
            )

        current.append(piece)
        current_len += join_len + piece_len

    if current:
        chunk = joiner.join(current)
        if chunk.strip():
            chunks.append(chunk)

    return chunks if chunks else [text]


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80) -> None:
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be > chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            raise ChunkingError(f"Empty document: {document.doc_id}")
        pieces = _split_text(
            document.content, _DEFAULT_SEPARATORS, self.chunk_size, self.chunk_overlap
        )
        pieces = [p for p in pieces if p.strip()]
        if not pieces:
            raise ChunkingError(f"Splitter produced 0 chunks: {document.doc_id}")

        chunks: list[Chunk] = []
        for idx, piece in enumerate(pieces):
            chunk_id = hashlib.sha256(
                f"{document.doc_id}:{idx}:{piece}".encode()
            ).hexdigest()
            meta = dict(document.metadata)
            meta["source_path"] = document.source_path
            meta["chunk_index"] = idx
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=piece,
                    metadata=meta,
                )
            )
        return chunks


__all__ = ["RecursiveTextSplitter"]
