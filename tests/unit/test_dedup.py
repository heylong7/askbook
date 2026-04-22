from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askbook.core.models import Chunk


def _chunk(cid: str, doc_id: str = "d1", content: str = "x") -> "Chunk":
    from askbook.core.models import Chunk

    return Chunk(chunk_id=cid, doc_id=doc_id, content=content)


def test_dedup_first_ingest_everything_is_new() -> None:
    from askbook.ingestion.dedup import SHA256Deduplicator

    new_chunks = [_chunk("c1"), _chunk("c2")]
    new, stale = SHA256Deduplicator().filter_new_chunks(
        new_chunks=new_chunks,
        existing_chunk_ids=set(),
    )
    assert [c.chunk_id for c in new] == ["c1", "c2"]
    assert stale == []


def test_dedup_full_duplicate_all_reused() -> None:
    from askbook.ingestion.dedup import SHA256Deduplicator

    new_chunks = [_chunk("c1"), _chunk("c2")]
    new, stale = SHA256Deduplicator().filter_new_chunks(
        new_chunks=new_chunks,
        existing_chunk_ids={"c1", "c2"},
    )
    assert new == []
    assert stale == []


def test_dedup_partial_update_returns_new_and_stale() -> None:
    from askbook.ingestion.dedup import SHA256Deduplicator

    new_chunks = [_chunk("c1"), _chunk("c4")]
    new, stale = SHA256Deduplicator().filter_new_chunks(
        new_chunks=new_chunks,
        existing_chunk_ids={"c1", "c2", "c3"},
    )
    assert [c.chunk_id for c in new] == ["c4"]
    assert sorted(stale) == ["c2", "c3"]


def test_dedup_file_deleted_all_stale() -> None:
    from askbook.ingestion.dedup import SHA256Deduplicator

    new, stale = SHA256Deduplicator().filter_new_chunks(
        new_chunks=[],
        existing_chunk_ids={"c1", "c2"},
    )
    assert new == []
    assert sorted(stale) == ["c1", "c2"]
