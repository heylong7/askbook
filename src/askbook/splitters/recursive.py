"""Pure-Python RecursiveCharacterTextSplitter — algorithm-compatible with langchain."""

from __future__ import annotations

import hashlib
import re

from askbook.core.exceptions import ChunkingError
from askbook.core.models import Chunk, Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_text_with_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: type[str] = str,  # kept for signature compat; unused
) -> list[str]:
    """Recursively split *text* using the first separator that works, then merge."""
    # Pick the first separator that actually splits this text
    separator = separators[-1]  # fallback: character-level
    new_separators: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            break
        if re.search(re.escape(sep), text):
            separator = sep
            new_separators = separators[i + 1 :]
            break

    splits = re.split(re.escape(separator), text) if separator else list(text)

    # Recursively process pieces that are still too large
    good_splits: list[str] = []
    for s in splits:
        if len(s) <= chunk_size:
            good_splits.append(s)
        else:
            if new_separators:
                good_splits.extend(
                    _split_text_with_separators(
                        s, new_separators, chunk_size, chunk_overlap
                    )
                )
            else:
                good_splits.append(s)

    # Merge small pieces into chunks respecting chunk_size / chunk_overlap
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _join(parts: list[str]) -> str:
        return separator.join(p for p in parts if p)

    for piece in good_splits:
        piece_len = len(piece)
        sep_len = len(separator) if current else 0
        if current_len + sep_len + piece_len > chunk_size and current:
            merged = _join(current)
            if merged.strip():
                chunks.append(merged)
            # Slide window: drop leading pieces until we're within overlap budget
            while current and current_len > chunk_overlap:
                dropped = current.pop(0)
                current_len -= len(dropped) + (len(separator) if current else 0)
            current_len = len(_join(current))
        current.append(piece)
        current_len = len(_join(current))

    if current:
        merged = _join(current)
        if merged.strip():
            chunks.append(merged)

    return chunks or [text]


class RecursiveCharacterTextSplitter:
    """Drop-in replacement for langchain's RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be > chunk_overlap")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = separators if separators is not None else _DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        return _split_text_with_separators(
            text, self._separators, self.chunk_size, self.chunk_overlap
        )

    def split(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            raise ChunkingError(f"Empty document: {document.doc_id}")

        pieces = [p for p in self.split_text(document.content) if p.strip()]
        if not pieces:
            raise ChunkingError(f"Splitter produced 0 chunks: {document.doc_id}")

        chunks: list[Chunk] = []
        for idx, piece in enumerate(pieces):
            chunk_id = hashlib.sha256(
                f"{document.doc_id}:{idx}:{piece}".encode()
            ).hexdigest()
            meta = {
                **document.metadata,
                "source_path": document.source_path,
                "chunk_index": idx,
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=piece,
                    metadata=meta,
                )
            )
        return chunks


# Backward-compat alias used in existing code
RecursiveTextSplitter = RecursiveCharacterTextSplitter

__all__ = ["RecursiveCharacterTextSplitter", "RecursiveTextSplitter"]
