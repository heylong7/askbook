# MCP Server 与工具协议

askbook 以 MCP（Model Context Protocol）stdio 模式暴露知识库，Claude Desktop 等 AI 助手可通过标准化工具调用直接查询私有文档。

## 架构

服务端基于 Python MCP SDK，通过 stdio 传输。stdout 留给 JSON-RPC 消息，日志全部输出到 stderr。启动方式：

```bash
uv run askbook serve --collection <name> [--config <path>]
```

## 七个工具

服务端注册七个工具，分为核心工具和诊断工具两类：

### 核心工具

| 工具 | 功能 | 输入 |
|------|------|------|
| **search** | 关键词+语义混合搜索，返回排序片段及 chunk_id | query, collection, top_k (1-20) |
| **ask** | 完整 RAG 问答，返回 LLM 合成答案及引用来源 | question, collection, top_k (1-10) |
| **get_chunk_content** | 按 chunk_id 获取完整 chunk 内容 | chunk_id, collection（可选） |
| **list_collections** | 列出所有 collection 及 chunk 数 | collection（可选） |
| **get_document_summary** | 按 doc_id 查看文档摘要（chunk 数、来源路径、内容预览） | doc_id, collection（可选） |

### 诊断工具

| 工具 | 功能 | 输入 |
|------|------|------|
| **collection_stats** | Collection 详细统计（chunk 数、文档数、磁盘大小） | collection（可选） |
| **trace_lookup** | 按 trace_id 查询链路事件，或列出最近事件 | trace_id（可选）, limit, days |

## ToolResponse 信封

所有工具返回统一的 `ToolResponse` 结构：

```json
{
  "status": "success",
  "summary": "找到 3 条关于 'RAG' 的结果。",
  "data": {},
  "source_ids": ["chunk_1", "chunk_2"],
  "next_actions": []
}
```

pydantic model_validator 强制要求：`status=success` 必须附带非空 `source_ids`。检索无结果时必须返回 `status=warning`，防止 LLM 在无依据的情况下生成幻觉答案。

## Claude Desktop 集成

在 Claude Desktop 的 MCP 设置中添加：

```json
{
  "mcpServers": {
    "askbook": {
      "command": "uv",
      "args": ["run", "askbook", "serve", "--collection", "docs"]
    }
  }
}
```

重启 Claude Desktop 后即可在对话中使用 askbook 工具。
