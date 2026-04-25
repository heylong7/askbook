# askbook

本地运行的私有知识库 RAG 系统 + MCP Server。

[![CI](https://github.com/<owner>/askbook/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/askbook/actions/workflows/ci.yml)

## 快速开始

```bash
uv sync --all-extras
cp .env.example .env
askbook --help
```

详细开发规范见 `DEV_SPEC.md`。

## MCP 接入 Claude Desktop

1. 先通过 `askbook ingest` 建立至少一个 collection。
2. 将 `examples/claude_desktop_mcp.json` 内容合并进 Claude Desktop 配置。
3. 重启 Claude Desktop，在对话中调用 `search` / `ask` / `list_collections` / `get_document_summary`。
