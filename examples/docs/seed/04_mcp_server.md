# MCP Server and Tool Contracts

askbook exposes its knowledge base as an MCP (Model Context Protocol) server running in stdio mode. This allows AI assistants like Claude Desktop and GitHub Copilot to directly query private documents through standardized tool calls.

## Architecture

The MCP server uses Python's MCP SDK with stdio transport. stdout is reserved for JSON-RPC messages; all logging goes to stderr. The server is started via `askbook serve --collection <name>`.

## Six Tools

The server registers six tools, following the principle of minimal but sufficient surface area:

### Core Tools

1. **search** -- Semantic search returning ranked snippets. Input: query, collection, top_k (1-20).
2. **ask** -- Full RAG Q&A with LLM-synthesized answer and citations. Input: question, collection, top_k (1-10).
3. **list_collections** -- List all available Chroma collections.
4. **get_document_summary** -- View summary metadata for a specific document by doc_id.

### Diagnostic Tools

5. **explain_retrieval** -- Show which chunks were retrieved and why, for debugging retrieval quality.
6. **health_check** -- Return health status of LLM, ChromaDB, and BM25 index components.

## ToolResponse Envelope

All tools return a consistent `ToolResponse` structure:

```json
{
  "status": "success",
  "summary": "Found 3 result(s) for 'RAG'.",
  "data": {},
  "source_ids": ["chunk_1", "chunk_2"],
  "next_actions": []
}
```

A pydantic model_validator enforces that `status=success` requires non-empty `source_ids`. If retrieval yields no results, the tool must return `status=warning`. This prevents hallucinated completions where the LLM fabricates answers without source support.

## Claude Desktop Integration

Add the server configuration to Claude Desktop's MCP settings file. The server command uses `uv run askbook serve --collection demo`. Restart Claude Desktop to see the six askbook tools available in conversations.
