# Configuration System

askbook uses a three-tier configuration strategy: YAML files for structured settings, `.env` for secrets, and pydantic-settings for startup validation.

## Three-Tier Strategy

1. **YAML configuration files** define all non-secret settings: LLM provider, model names, chunk sizes, collection names, and pipeline parameters. Multiple YAML files support different environments (local Ollama vs cloud DashScope).
2. **`.env` file** injects secrets like API keys as environment variables. This file is gitignored and never committed.
3. **pydantic-settings** validates all configuration at startup with strict type checking. Invalid configs fail fast with field-level error messages.

## Environment Variable Override

Environment variables use the prefix `ASKBOOK_` with double-underscore nesting:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASKBOOK_LLM__PROVIDER` | `ollama` | LLM provider (ollama, dashscope, openai) |
| `ASKBOOK_EMBEDDING__PROVIDER` | `bge-m3` | Embedding model provider |
| `ASKBOOK_VECTORSTORE__PATH` | `~/.askbook/chroma` | ChromaDB storage path |
| `ASKBOOK_DATA_DIR` | `~/.askbook` | BM25 index and runtime data directory |

## Configuration Sections

The main `default.yaml` covers eight configuration domains:

- **llm**: provider, model, temperature, max_tokens, fallback chain, quota limits
- **embedding**: model name, normalization, device (cpu/cuda/mps), batch size
- **reranker**: coarse Cross-Encoder model, fine LLM rerank toggle
- **vectorstore**: provider (chroma), persist directory, schema version
- **ingestion**: splitter type, chunk_size (512), chunk_overlap (64), enrichment toggles, concurrency limits
- **query**: top_k, BM25/dense RRF weights, rewriter and HyDE toggles
- **mcp**: allowed path whitelist, max results per query
- **observability**: trace mode (async/sync), PII redaction patterns, retention days, dashboard port

## Provider Fallback Chain

The LLM configuration supports a fallback chain for high availability. If the primary provider times out or returns an error, the system automatically tries the next provider in the chain. For example: primary DashScope with fallback to local Ollama.
