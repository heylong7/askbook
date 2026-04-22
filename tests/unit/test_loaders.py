from pathlib import Path

import pytest


def test_loader_produces_document_with_deterministic_doc_id(tmp_path: Path) -> None:
    from askbook.ingestion.loaders import MarkItDownLoader

    src = tmp_path / "sample.md"
    src.write_text("# Title\n\nBody.\n", encoding="utf-8")

    loader = MarkItDownLoader()
    doc = loader.load(src)

    assert doc.source_path == str(src.resolve())
    assert doc.doc_id  # non-empty
    assert "Body" in doc.content
    # Same file, same mtime → same doc_id
    assert loader.load(src).doc_id == doc.doc_id


def test_loader_raises_document_load_error_for_unsupported(tmp_path: Path) -> None:
    from askbook.core.exceptions import DocumentLoadError
    from askbook.ingestion.loaders import MarkItDownLoader

    src = tmp_path / "unsupported.xyz"
    src.write_bytes(b"\x00\x01\x02")

    with pytest.raises(DocumentLoadError):
        MarkItDownLoader().load(src)


def test_loader_iter_files_recurses(tmp_path: Path) -> None:
    from askbook.ingestion.loaders import MarkItDownLoader

    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("plain", encoding="utf-8")

    files = list(MarkItDownLoader.iter_files(tmp_path))
    names = sorted(p.name for p in files)
    assert names == ["a.md", "b.txt"]
