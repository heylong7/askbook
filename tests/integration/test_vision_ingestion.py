"""Integration tests for LLMEnrichmentNode."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from askbook.core.models import Chunk
from askbook.ingestion.enrichment import LLMEnrichmentNode, _image_to_data_uri
from askbook.observability.null_trace import NullTraceWriter
from askbook.providers.stub import StubLLMProvider


@pytest.fixture
def stub_llm():
    return StubLLMProvider(model="test")


@pytest.fixture
def trace_writer():
    return NullTraceWriter()


@pytest.fixture
def img_dir(tmp_path):
    """Create a temp dir with a minimal valid PNG file."""

    def _create_png(path: Path) -> None:
        sig = b"\x89PNG\r\n\x1a\n"
        # IHDR chunk — 1x1 pixel, 8-bit RGB
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        # IDAT chunk
        raw = zlib.compress(b"\x00\xff\x00\xff")
        idat_crc = zlib.crc32(b"IDAT" + raw)
        idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
        # IEND chunk
        iend_crc = zlib.crc32(b"IEND")
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        path.write_bytes(sig + ihdr + idat + iend)

    img = tmp_path / "test.png"
    _create_png(img)
    return tmp_path


class TestLLMEnrichmentNode:
    def test_passthrough_chunks_without_images(self, stub_llm, trace_writer):
        """Chunks without image references pass through unchanged."""
        node = LLMEnrichmentNode(llm=stub_llm, trace_writer=trace_writer)
        chunk = Chunk(
            chunk_id="c1",
            content="# Hello\n\nThis is text only.",
            doc_id="doc1",
            metadata={"source_path": "/tmp/doc.md"},
        )
        result = node.run({"chunks": [chunk], "source_path": "/tmp/doc.md"})
        assert len(result["chunks"]) == 1
        assert result["chunks"][0].content == chunk.content

    def test_extracts_markdown_image_and_calls_llm(
        self, stub_llm, trace_writer, img_dir
    ):
        """Image ref in chunk -> LLM called once, description appended."""
        md_file = img_dir / "doc.md"
        md_file.write_text("# Doc with image")
        node = LLMEnrichmentNode(llm=stub_llm, trace_writer=trace_writer)
        chunk = Chunk(
            chunk_id="c1",
            content="Some text\n![alt](test.png)\nMore text",
            doc_id="doc1",
            metadata={"source_path": str(md_file)},
        )
        result = node.run({"chunks": [chunk], "source_path": str(md_file)})
        enriched = result["chunks"][0]
        assert "Image Descriptions" in enriched.content
        assert "stub" in enriched.content.lower()  # StubLLM returns canned content

    def test_skips_missing_image_file(self, stub_llm, trace_writer, img_dir):
        """Missing image file -> log warning, chunk unchanged."""
        md_file = img_dir / "doc.md"
        md_file.write_text("# Doc")
        node = LLMEnrichmentNode(llm=stub_llm, trace_writer=trace_writer)
        chunk = Chunk(
            chunk_id="c1",
            content="![alt](nonexistent.png)",
            doc_id="doc1",
            metadata={"source_path": str(md_file)},
        )
        result = node.run({"chunks": [chunk], "source_path": str(md_file)})
        # Should not crash; chunk content unchanged (no Image Descriptions section)
        assert "Image Descriptions" not in result["chunks"][0].content

    def test_llm_none_passthrough(self, trace_writer):
        """When llm is None the node passes chunks through unchanged."""
        node = LLMEnrichmentNode(llm=None, trace_writer=trace_writer)
        chunk = Chunk(
            chunk_id="c1",
            content="![alt](img.png)",
            doc_id="doc1",
        )
        result = node.run({"chunks": [chunk], "source_path": "/tmp/doc.md"})
        assert result["chunks"][0].content == chunk.content


def test_image_to_data_uri_readable(img_dir):
    """_image_to_data_uri returns a valid data URI for an existing file."""
    uri = _image_to_data_uri(img_dir / "test.png")
    assert uri is not None
    assert uri.startswith("data:image/png;base64,")


def test_image_to_data_uri_none_for_missing():
    """_image_to_data_uri returns None for a non-existent file."""
    assert _image_to_data_uri(Path("/nonexistent/img.png")) is None
