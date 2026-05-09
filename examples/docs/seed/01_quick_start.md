# 快速开始

askbook 是一个本地私有知识库 RAG 系统 + MCP Server。支持多种 LLM 和 Embedding 后端，通过 stdio 模式暴露 MCP 工具供 Claude Desktop 等 AI 助手直接调用。

## 环境要求

Python 3.12+、uv 包管理器。如需本地模型还需安装 Ollama。

## 安装

```bash
git clone <repo>
cd askbook
uv sync --all-extras
```

配置 API Key（使用百炼或 OpenAI 时需要）：

```bash
export DASHSCOPE_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## 基本用法

### 1. 入库文档

```bash
# 入库单个文件到指定 collection
uv run askbook ingest path/to/doc.md --collection my-docs

# 入库整个目录（支持 .md / .txt / .pdf / .docx）
uv run askbook ingest docs/ --collection my-docs

# 指定配置文件
uv run askbook ingest docs/ --collection my-docs --config configs/bailian.yaml

# 强制重建索引
uv run askbook ingest docs/ --collection my-docs --force-reindex
```

### 2. 命令行查询

```bash
# 直接提问
uv run askbook query "什么是 RAG？" --collection my-docs

# 调整返回数量
uv run askbook query "如何评估检索质量？" --collection my-docs --top-k 10
```

### 3. 启动 MCP Server

```bash
# stdio 模式（供 Claude Desktop 调用）
uv run askbook serve --collection my-docs

# 指定配置
uv run askbook serve --collection my-docs --config configs/bailian.yaml
```

### 4. 其他命令

```bash
# 启动 Streamlit 看板
uv run askbook dashboard

# 运行评估
uv run askbook eval --collection my-docs

# 垃圾回收（清理孤立 chunk、过期 trace）
uv run askbook gc
```

### 5. 开发

```bash
# 运行测试
uv run pytest -q

# 完整质量检查
uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/ && uv run pytest --cov=src/askbook --cov-fail-under=80 -q
```
