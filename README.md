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

## 快速开始

```bash
# 1. 安装
pip install askbook

# 2. 摄入文档
askbook ingest ./docs --collection demo

# 3. 提问
askbook query "什么是 RAG？" --collection demo
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
| `ASKBOOK_LLM__PROVIDER` | `ollama` | LLM 提供商（`ollama` / `dashscope` / `stub`） |
| `ASKBOOK_EMBEDDING__PROVIDER` | `bge-m3` | Embedding 模型（`bge-m3` / `dashscope` / `stub`） |
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
   - `trace_lookup` — 按 trace_id 查询链路事件或浏览近期事件
   - `collection_stats` — 查看 collection 的详细统计（chunk 数、文档数、磁盘大小）

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

详细开发规范见 `DEV_SPEC.md`。当前进度：Phase 7 (v1.0) ✅ 完成，379 个测试，覆盖率 ≥ 80%。

## 可观测性（Trace + Dashboard）

askbook 内置异步 JSONL Trace，无需外部 APM。

### Trace 写入

每次运行 `ingest` / `query` / MCP 调用时，自动将 span 事件追加到：

```
~/.askbook/traces/YYYY-MM-DD.jsonl
```

### 查看 Dashboard

```bash
uv run askbook dashboard          # 使用默认端口 8501
uv run askbook dashboard --port 9000
```

浏览器打开 `http://localhost:8501`，可查看：
- **页面 1 — 系统总览**：今日 Query 数 / P50 延迟 / Token 总量 / 各组件健康状态
- **页面 2 — 数据浏览**：Chroma Collection 文档列表与 Chunk 预览
- **页面 3 — Ingestion 监控**：Ingestion 运行记录 + Plotly Gantt 甘特图 + Query Rewrite 审计
- **页面 4 — Trace 查看器**：Trace 事件浏览、筛选与详情查看
- **页面 5 — 评估与质量**：检索指标（Hit Rate / MRR / Recall / NDCG）、Ragas 与 LLM-judge 评分、Harness 健康卡片、用户反馈收集

### 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `observability.enabled` | `true` | 关闭后所有写入操作短路，恢复 NullTraceWriter |
| `observability.trace_dir` | `~/.askbook/traces` | JSONL 文件存储目录 |
| `observability.retention_days` | `7` | 自动清理超过此天数的 trace 文件 |
| `observability.pii_redaction` | `true` | 写入前对手机号 / 邮箱 / token 串脱敏 |
| `observability.dashboard_port` | `8501` | Streamlit 默认端口 |

### Harness 30.1.3 说明

`QuerySpan.original_query` 是必填字段，由 `QueryRewriterNode` 在 passthrough 和改写两种模式下均保证写入，为 Phase 6 Rewrite Diff 视图提供数据支撑。

## Evaluation (v0.1)

`askbook` ships with a 20-item human-curated QA set at `datasets/seed_manual.yaml`
and the following evaluation metrics:

**Retrieval metrics** (4 core metrics):
- `hit_rate`、`mrr`、`recall@k`、`ndcg@k`

**Answer quality metrics** (Ragas):
- `faithfulness` — 答案是否忠实于检索上下文
- `answer_relevancy` — 答案与问题的相关性

**LLM-judge** (parallel judge evaluation):
- 使用 LLM-as-judge 模式对每对 QA 进行 faithfulness 与 relevancy 评分
- 基于 asyncio.Semaphore 实现并发判卷，支持自定义并发度

Run the suite once your sample collection is ingested:

```bash
uv run askbook ingest examples/docs/seed --collection demo
uv run askbook eval --dataset datasets/seed_manual.yaml --collection demo --k 5
```

Each run writes `eval_runs/<timestamp>.json`. To establish a new baseline (only
when you intentionally improved retrieval), pass `--update-baseline
tests/golden/baselines/v0.1_scores.json`. CI runs `pytest -m golden` and fails
when any metric drops more than 0.05 below the recorded baseline.

## Harness 健康指标

askbook 内置 Harness 健康监控，通过 Dashboard 页面 5 实时呈现：

| 指标 | 目标 | 说明 |
|------|------|------|
| **completion_rate** | >= 95% | MCP 工具调用成功率 |
| **retries_per_task** | <= 1.2 | 每个任务的平均重试次数 |
| **pass@1** | >= 85% | Golden 数据集首轮通过率 |
| **cost_per_task** | <= ¥0.05 | 每次成功 ask 调用的平均成本 |

**Anti-pattern CI 检查**：Harness 层对 6 种反模式做静态分析（详见 DEV_SPEC §30.3），包括 context rot、hallucinated completion、model drift 等，确保每次代码提交符合质量门禁。
