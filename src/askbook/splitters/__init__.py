"""askbook.splitters — chunking strategies for document ingestion."""

from askbook.splitters.recursive import (
    RecursiveCharacterTextSplitter,
    RecursiveTextSplitter,
)
from askbook.splitters.semantic import SemanticSplitter

__all__ = [
    "RecursiveCharacterTextSplitter",
    "RecursiveTextSplitter",
    "SemanticSplitter",
]
