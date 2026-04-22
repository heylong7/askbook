"""SHA256 chunk-level deduplicator (DEV_SPEC Ch 20 R7)."""

from __future__ import annotations

from askbook.core.models import Chunk


class SHA256Deduplicator:
    def filter_new_chunks(
        self,
        new_chunks: list[Chunk],
        existing_chunk_ids: set[str],
    ) -> tuple[list[Chunk], list[str]]:
        """Return (chunks_to_add, stale_chunk_ids_to_delete).

        - chunks_to_add: in new_chunks but not in existing_chunk_ids
        - stale_chunk_ids: in existing_chunk_ids but not in new_chunks
        """
        new_ids = {c.chunk_id for c in new_chunks}
        to_add = [c for c in new_chunks if c.chunk_id not in existing_chunk_ids]
        stale = sorted(existing_chunk_ids - new_ids)
        return to_add, stale


__all__ = ["SHA256Deduplicator"]
