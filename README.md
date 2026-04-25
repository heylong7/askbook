# askbook

本地运行的私有知识库 RAG 系统 + MCP Server。支持 Ollama 本地 LLM，以 stdio 模式暴露 MCP 工具供 Claude Desktop 直接调用。

[![CI](https://github.com/<owner>/askbook/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/askbook/actions/workflows/ci.yml)

## 安装

```bash
# 依赖管理用 uv（需 Python 3.12+）
uv sync --all-extras

# 复制配置模板（按需填写 Ollama 地址等）
cp .env.example .env
```

## 使用方法

### 1. 导入文档

```bash
# 导入单个文件，指定 collection 名称
askbook ingest path/to/doc.pdf --collection demo

# 导入整个目录（递归，支持 .pdf / .txt / .md）
askbook ingest docs/ --collection my-notes

# 查看所有可用 collection
askbook ingest --list
```

### 2. 命令行查询

```bash
# 语义搜索（返回片段）
askbook query search "什么是 RAG？" --collection demo

# 问答（走完整 Pipeline，LLM 合成答案 + 引用来源）
askbook query ask "如何评估检索质量？" --collection demo

# 查看 collection 信息
askbook query list-collections
```

### 3. 启动 MCP Server

```bash
# 以 stdio 模式启动，供 Claude Desktop 调用
askbook serve --collection demo

# 指定配置文件（默认读取环境变量）
askbook serve --collection demo --config ~/.askbook/config.yaml
```

## 配置

通过环境变量覆盖默认值：

| 变量 | 默认 | 说明 |
|------|------|------|
| `ASKBOOK_LLM__PROVIDER` | `ollama` | LLM 提供商（`ollama` / `stub`） |
| `ASKBOOK_EMBEDDING__PROVIDER` | `bge-m3` | Embedding 模型（`bge-m3` / `stub`） |
| `ASKBOOK_VECTORSTORE__PATH` | `~/.askbook/chroma` | ChromaDB 存储路径 |
| `ASKBOOK_DATA_DIR` | `~/.askbook` | BM25 索引等数据目录 |

## MCP 接入 Claude Desktop

先确保 Ollama 在本机运行（`ollama serve`），并已拉取模型（`ollama pull qwen2.5:7b`）。

**步骤：**

1. 导入至少一个 collection：`askbook ingest docs/ --collection demo`
2. 将 `examples/claude_desktop_mcp.json` 内容合并进 Claude Desktop 的 MCP 配置文件。
3. 重启 Claude Desktop。在对话中可调用以下工具：
   - `search` — 语义搜索，返回排序片段
   - `ask` — 完整 RAG 问答，返回答案 + 引用来源
   - `list_collections` — 列出所有可用 collection
   - `get_document_summary` — 查看指定文档的摘要信息

**工具返回格式（`ToolResponse`）：**

```json
{
  "status": "success",        // success / warning / error
  "summary": "Found 3 result(s) for 'RAG'.",
  "data": { ... },            // 工具特有数据
  "source_ids": ["chunk_1"],  // success 时必须非空（Harness 30.1.2）
  "next_actions": []          // warning/error 时的建议操作
}
```

## 开发

```bash
# 运行全量测试
uv run pytest -q

# 质量门禁（ruff + mypy + pytest + 覆盖率）
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/ && uv run pytest --cov=src/askbook --cov-fail-under=80 -q
```

详细开发规范见 `DEV_SPEC.md`。当前进度：Phase 3（MCP Server）✅ 完成，182 个测试，83.74% 覆盖率。
