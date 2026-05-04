# System Architecture

askbook is a modular, pluggable RAG system with clean separation between ingestion (write path) and query (read path). All external dependencies (LLM, Embedder, VectorStore, Reranker) are abstracted behind Protocols, enabling provider swaps through configuration changes alone.

## Core Design Principles

### Pluggable Abstraction Layer

Six Protocols (structural subtyping) and two ABCs (inheritance-based) define the system's extension points:

- **LLMProviderProtocol**: complete() and acomplete() for text generation
- **EmbedderProtocol**: embed_query(), embed_passage(), embed_batch()
- **RerankerProtocol**: rerank() for relevance scoring
- **SplitterProtocol**: split() for document chunking
- **TraceWriterProtocol**: span() context manager for observability
- **VectorStoreABC**: 5 abstract methods for vector storage operations
- **BasePipelineNode**: template method with built-in trace auto-reporting
- **BaseEvaluator**: template method with result serialization

New providers are added by implementing the corresponding Protocol and registering in the ServiceRegistry factory. No existing code changes required.

### Data Flow Separation

The ingestion pipeline (write path) and query pipeline (read path) are completely independent. Ingestion runs as a synchronous CLI batch process. Query runs as an online inference pipeline triggered by MCP tool calls or CLI commands.

### Directory Structure

Source code follows a `src/` layout organized by functional domain:
- `core/` -- interfaces, models, exceptions, registry
- `config/` -- settings, schema, defaults
- `providers/` -- LLM adapter implementations
- `embeddings/` -- embedding model wrappers
- `rerankers/` -- Cross-Encoder and LLM reranker
- `vectorstores/` -- ChromaDB adapter
- `ingestion/` -- 7-node document ingestion pipeline
- `query/` -- 7-node query processing pipeline
- `mcp_server/` -- MCP stdio server and tool handlers
- `observability/` -- async JSONL trace writer
- `evaluation/` -- dataset loader, runner, metrics
- `dashboard/` -- Streamlit multi-page app

## Runtime Data Layout

All runtime artifacts live under `~/.askbook/`:
- `chroma/` -- ChromaDB persistent storage
- `traces/` -- JSONL trace logs (sharded by date)
- `cache/` -- MarkItDown and HuggingFace model caches
- `eval_runs/` -- evaluation result snapshots
