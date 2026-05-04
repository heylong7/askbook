# Document Ingestion Pipeline

The Ingestion Pipeline transforms raw documents (PDF, Markdown, DOCX, TXT) into searchable chunks stored in ChromaDB and a BM25 index. It runs as a synchronous CLI process with a 7-node pipeline.

## Pipeline Nodes

```
Input: source_path
  |
[DocumentLoaderNode]  -> Document (MarkItDown converts to Markdown)
  |
[SplitterNode]        -> list[Chunk] (recursive character split with overlap)
  |
[EnrichmentNode]      -> list[Chunk] (image descriptions stitched in, metadata injected)
  |
[DedupNode]           -> (new_chunks, stale_chunk_ids)
  |
[EmbeddingNode]       -> list[Chunk] (BGE-M3 dense embeddings)
  |
[VectorStoreWriteNode] -> upsert new chunks, delete stale chunks from Chroma
  |
[BM25IndexUpdateNode] -> update persistent BM25 pickle index
  |
Output: IngestionResult
```

## SHA256-Based Deduplication

The deduplicator computes a SHA256 hash of each chunk's content. Chunks that already exist in the target collection are skipped (reused). Chunks from a previously ingested version of the same source file that no longer appear in the new version are flagged as stale and soft-deleted from ChromaDB. This prevents zombie data accumulation.

## Key Design Decisions

- **doc_id** is SHA256 of (absolute path + mtime_ns), ensuring idempotent re-ingestion.
- **chunk_id** is SHA256 of (doc_id + chunk_index + content), guaranteeing unique identification.
- **Soft delete** queries existing chunks by source_path, computes the diff, and removes only stale entries.
- **Enrichment** passes through by default in v0.1; Vision LLM image description is planned for v0.5.
- **Concurrency** uses three independent semaphores for embedding, vision LLM, and Chroma writes.
