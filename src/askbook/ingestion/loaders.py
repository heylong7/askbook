"""Document loaders — single MarkItDown-backed implementation with suffix dispatch."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from markitdown import MarkItDown

from askbook.core.exceptions import DocumentLoadError
from askbook.core.models import Document

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".pdf", ".docx", ".html", ".htm"}
)


class MarkItDownLoader:
    """Loads a single file into a Document via MarkItDown."""

    def __init__(self) -> None:
        self._md = MarkItDown()

    def load(self, path: Path) -> Document:
        p = Path(path).resolve()
        suffix = p.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise DocumentLoadError(f"Unsupported file type: {p.name}")
        if not p.is_file():
            raise DocumentLoadError(f"Not a file: {p}")
        try:
            result = self._md.convert(str(p))
        except Exception as e:
            raise DocumentLoadError(f"MarkItDown failed on {p.name}: {e}") from e

        content: str = getattr(result, "text_content", None) or ""
        if not content.strip():
            raise DocumentLoadError(f"Empty content after conversion: {p.name}")

        mtime = p.stat().st_mtime_ns
        doc_id = hashlib.sha256(f"{p}|{mtime}".encode()).hexdigest()
        return Document(
            doc_id=doc_id,
            source_path=str(p),
            content=content,
            metadata={"suffix": suffix, "mtime_ns": mtime},
        )

    @staticmethod
    def iter_files(root: Path) -> Iterator[Path]:
        root = Path(root)
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_SUFFIXES:
                yield root
            return
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                yield p


__all__ = ["MarkItDownLoader", "SUPPORTED_SUFFIXES"]
