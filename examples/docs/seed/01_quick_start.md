# Quick Start

askbook is a private knowledge base RAG system + MCP Server that runs locally. It supports Ollama local LLM and exposes MCP tools via stdio mode for direct invocation by Claude Desktop.

## Installation

Requirements: Python 3.12+, uv package manager, and Ollama running locally.

```bash
uv sync --all-extras
cp .env.example .env
```

Edit `.env` to configure your Ollama host address and model preferences. The default configuration uses `ollama` as the LLM provider and `bge-m3` as the embedding model, both running locally at zero API cost.

## Basic Usage

### 1. Import Documents

```bash
# Import a single file into a named collection
askbook ingest path/to/doc.pdf --collection demo

# Import an entire directory recursively (supports .pdf / .txt / .md)
askbook ingest docs/ --collection my-notes

# List all available collections
askbook ingest --list
```

### 2. Command-Line Query

```bash
# Semantic search returning snippets
askbook query search "what is RAG?" --collection demo

# Full RAG Q&A with LLM-synthesized answers and citations
askbook query ask "how to evaluate retrieval quality?" --collection demo

# View collection information
askbook query list-collections
```

### 3. Start MCP Server

```bash
# Start in stdio mode for Claude Desktop
askbook serve --collection demo

# Specify a custom config file
askbook serve --collection demo --config ~/.askbook/config.yaml
```

### 4. Development

```bash
# Run all tests
uv run pytest -q

# Quality gate (ruff + mypy + pytest + coverage >= 80%)
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/ && uv run pytest --cov=src/askbook --cov-fail-under=80 -q
```
