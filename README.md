
<div align="center">
<img src="assets/icon.png" alt="askbook icon" width="520">
</div>

本地优先的私有知识库 RAG 系统——把文档导入后，用自然语言提问，获得带原文引用的准确回答。数据全程不出机器。

**核心特点：**

- **本地优先** — ChromaDB + Ollama + BGE-M3 全本地运行，零云端依赖
- **全链路可观测** — 每次调用自动写 JSONL trace，Dashboard 5 页监控，PII 脱敏
- **MCP 原生** — 以 MCP Server 对外暴露，Claude Desktop / Claude Code 直接调用私有知识库
- **多 Provider** — LLM 支持 Ollama / DashScope / OpenAI / Anthropic，Embedding 支持 BGE-M3 / OpenAI / DashScope，一个 YAML 切换
- **混合检索** — Dense (向量) + Sparse (BM25) + RRF 融合 + Cross-Encoder 精排
- **Harness 质量门禁** — 4 项健康指标 + 6 种反模式 CI 检查 + 评估基线门禁

---

## 快速开始

**前置条件：** Python 3.12+、8GB+ 内存、10GB+ 磁盘

### 1. 安装 Ollama

- [下载 Ollama](https://ollama.com/download) 并安装
- 启动后在 `localhost:11434` 监听

### 2. 拉取模型

```bash
ollama pull qwen2.5:7b          # 对话模型 (~4.5G)，中文场景推荐
```

> 显存 >= 16G 可选 `qwen2.5:14b`；纯英文场景选 `llama3.1:8b`。

### 3. 安装 uv

```bash
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 4. 克隆并安装

```bash
git clone https://github.com/heylong7/askbook.git
cd askbook
uv sync --all-extras
cp .env.example .env
```

### 5. 导入文档

```bash
uv run askbook ingest examples/docs/seed --collection demo    # 试用示例数据
uv run askbook ingest /path/to/docs --collection my-docs      # 你的文档
uv run askbook ingest /path/to/single.pdf --collection my-docs
```

### 6. 提问

```bash
uv run askbook query "如何配置 Ollama？" --collection demo
uv run askbook query "混合检索原理" --collection demo
```

---

## 架构与全链路

```
你的文档 (PDF / DOCX / MD / TXT / 图片)
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                      │
│                                                          │
│  load ──► split ──► enrich ──► dedup ──► embed ──► write│
│  (解析)    (分块)   (图片描述)  (去重)    (向量化)  (入库) │
│                                                          │
└──────────────────────────────────────────────────────────┘
  │                              │
  ▼                              ▼
┌──────────┐             ┌─────────────┐
│ ChromaDB │             │ BM25 Index  │
│(向量检索) │             │ (关键词检索) │
└──────────┘             └─────────────┘
  │                              │
  ▼                              ▼
┌──────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                         │
│                                                          │
│  rewrite ──► retrieve (dense + bm25 + RRF) ──► rerank    │
│  (改写)        (混合检索 + 融合排序)               (精排)   │
│                                                          │
│  ──► synthesize (LLM 合成答案 + 引用来源)                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
  │
  ▼
├── CLI:     uv run askbook query "xxx" --collection demo
├── MCP:     uv run askbook serve --collection demo
└── Web:     uv run askbook dashboard (Streamlit)
```

### Ingestion Pipeline（7 节点）

| 节点 | 功能 |
|------|------|
| **load** | MarkItDown 解析 PDF/DOCX/MD/TXT，转换为 Markdown |
| **split** | 递归字符分割，可切换语义分割 |
| **enrich** | 扫描 `![alt](path)` 图片引用，将图片发多模态 LLM 描述，追加到 chunk |
| **dedup** | SHA256 去重，跳过已索引的 chunk |
| **embed** | 调用 Embedder 向量化，支持并发 |
| **write** | 写入 ChromaDB，同步清理过期 chunk |
| **bm25** | 更新本地 BM25 关键词索引 |

### Query Pipeline（5 阶段）

| 阶段 | 功能 |
|------|------|
| **rewrite** | LLM 改写用户问题，补全上下文（可配置开关） |
| **HyDE** | 生成假设文档用于检索（可配置开关） |
| **retrieve** | 并行 dense + BM25，RRF 融合排序 |
| **rerank** | Cross-Encoder 精排，过滤低相关结果 |
| **synthesize** | LLM 合成答案 + 引用来源 chunk |

---

## 文档摄入

### 支持格式

通过 MarkItDown 统一转换为 Markdown：

| 格式 | 扩展名 |
|------|--------|
| PDF | `.pdf` |
| Word | `.docx` |
| Markdown | `.md` |
| 纯文本 | `.txt` |

### 分块策略

**默认：递归字符分割（RecursiveCharacterTextSplitter）**

按 `\n\n` → `\n` → ` ` → 字符 的优先级递归切分，保持语义完整性：

```yaml
ingestion:
  chunk_size: 600        # 每块最大字符数
  chunk_overlap: 80      # 相邻块重叠字符数
```

**可选：语义分割（SemanticSplitter）**

基于 embedding 相似度检测语义边界，更贴合文档的自然段落。适合长文档、结构化文档。

### 元数据增强

每个 chunk 创建时自动注入三层元数据：

```python
meta = {
    **document.metadata,      # 文档原始元数据（loader 提取）
    "source_path": "...",     # 来源文件路径
    "chunk_index": 0,         # 在文档中的位置序号
}
```

这些元数据随 chunk 存入 ChromaDB 和 BM25 索引，检索时随结果返回，用于溯源和引用。

### 图片处理

Enrichment 节点扫描 chunk 中的 Markdown 图片引用 `![alt](path)`，读取图片文件并转 base64 发给多模态 LLM 生成文字描述，追加到 chunk 内容末尾。

**需要配置 `enrich_llm`**——和主问答 LLM 分开，避免非多模态模型乱猜：

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6       # 问答用 Claude

ingestion:
  enrich_llm:                    # 图片描述用多模态模型
    provider: openai
    model: gpt-4o-mini
```

不配 `enrich_llm` 则跳过图片描述，文档无图片引用时也自动跳过。

---

## 检索与问答

### 混合检索

并行执行两条检索路径，RRF (Reciprocal Rank Fusion) 融合排序：

| 路径 | 技术 | 擅长 |
|------|------|------|
| **Dense** | ChromaDB 向量相似度 | 语义匹配、同义词 |
| **Sparse** | BM25 关键词索引 | 精确术语、专业词汇 |

### 重排序

Cross-Encoder BGE-Reranker-v2-m3 对融合结果精排，提升 Top-K 准确率：

```yaml
query:
  top_k: 10              # 粗排返回数
  rerank_top_k: 5        # 精排后保留数
  rrf_k: 60              # RRF 融合参数
```

### 查询增强

| 功能 | 配置 | 说明 |
|------|------|------|
| Query Rewrite | `query.enable_rewrite: true` | LLM 改写用户问题，补全上下文 |
| HyDE | `query.enable_hyde: true` | 生成假设文档，用假设文档向量检索 |
| LLM Rerank | `query.enable_llm_rerank: true` | 用 LLM 代替 Cross-Encoder 做精排 |

---

## 配置系统

### 优先级

```
环境变量 (ASKBOOK_*)  >  .env 文件  >  --config YAML  >  defaults.yaml（内置）
```

### Provider 选项

| 组件 | Provider | API Key 环境变量 | 常用模型 |
|------|----------|-----------------|----------|
| **LLM** | `ollama` | 无需 | `qwen2.5:7b` / `llama3:8b` / `deepseek-r1:8b` |
| **LLM** | `dashscope` | `DASHSCOPE_API_KEY` | `qwen-plus` / `qwen-max` |
| **LLM** | `openai` | `OPENAI_API_KEY` | `gpt-4o` / `gpt-4o-mini` / `o3-mini` |
| **LLM** | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` / `claude-opus-4-7` |
| **Embedding** | `bge-m3` | 无需 | `BAAI/bge-m3` (1024 维) |
| **Embedding** | `openai` | `OPENAI_API_KEY` | `text-embedding-3-small` / `text-embedding-3-large` |
| **Embedding** | `dashscope` | `DASHSCOPE_API_KEY` | `text-embedding-v3` |
| **Vector Store** | `chroma` | 无需 | 本地 ChromaDB |
| **Reranker** | `bge-v2-m3` | 无需 | BGE-Reranker-v2-m3 |

### 切换示例

```yaml
# 问答用 Claude，embedding 用 OpenAI
llm:
  provider: anthropic
  model: claude-sonnet-4-6

embedding:
  provider: openai
  model: text-embedding-3-small
```

```bash
export ANTHROPIC_API_KEY="sk-ant-xxx"
export OPENAI_API_KEY="sk-xxx"
uv run askbook serve --collection demo --config configs/default.yaml
```

### 完整配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm.provider` | `ollama` | LLM provider |
| `llm.model` | `qwen2.5:7b` | 模型名称 |
| `llm.temperature` | `0.1` | 生成温度 |
| `llm.max_tokens` | `4096` | 单次最大输出 token |
| `llm.fallback_chain` | `[]` | Fallback 链，主 provider 故障时自动降级 |
| `embedding.provider` | `bge-m3` | Embedding provider |
| `embedding.model` | `BAAI/bge-m3` | Embedding 模型名 |
| `embedding.device` | `auto` | 推理设备 (`cpu` / `cuda` / `auto`) |
| `embedding.batch_size` | `32` | 批处理大小 |
| `vectorstore.path` | `~/.askbook/chroma` | ChromaDB 存储路径 |
| `ingestion.chunk_size` | `600` | 分块大小（字符） |
| `ingestion.chunk_overlap` | `80` | 分块重叠（字符） |
| `ingestion.enrich_llm` | `null` | 图片描述专用 LLM，不设则跳过 |
| `query.top_k` | `10` | 粗排返回数 |
| `query.rerank_top_k` | `5` | 精排保留数 |
| `query.rrf_k` | `60` | RRF 融合参数 |
| `query.enable_rewrite` | `true` | 查询改写开关 |
| `query.enable_hyde` | `false` | HyDE 开关 |
| `query.enable_llm_rerank` | `false` | LLM 重排序开关 |
| `observability.retention_days` | `7` | Trace 保留天数 |
| `observability.pii_redaction` | `true` | PII 脱敏 |
| `data_dir` | `~/.askbook` | 数据根目录 |

---

## MCP Server（接入 Claude Code / Claude Desktop）

```bash
uv run askbook serve --collection demo
```

### 配置 Claude Desktop

将下面的配置合并到 Claude Desktop 的 MCP 配置文件中：

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "askbook": {
      "command": "uv",
      "args": ["run", "askbook", "serve", "--collection", "demo"],
      "cwd": "/path/to/askbook"
    }
  }
}
```

### 配置 Claude Code

在项目根目录创建 `.mcp.json`：

```json
{
  "mcpServers": {
    "askbook": {
      "command": "uv",
      "args": ["run", "askbook", "serve", "--collection", "demo"]
    }
  }
}
```

### 可用 MCP 工具（7 个）

| 工具 | 类型 | 说明 |
|------|------|------|
| `search` | 核心 | 关键词 + 语义混合搜索，返回排序片段 + chunk_id |
| `ask` | 核心 | 完整 RAG 问答，返回答案 + 引用来源 |
| `get_chunk_content` | 核心 | 按 chunk_id 获取完整 chunk 内容 |
| `list_collections` | 核心 | 列出所有 collection 及 chunk 数 |
| `get_document_summary` | 核心 | 指定文档的 chunk 数、来源路径、内容预览 |
| `collection_stats` | 诊断 | Collection 详细统计（chunk 数 / 文档数 / 磁盘大小） |
| `trace_lookup` | 诊断 | 按 trace_id 查询链路事件，或列出最近事件 |

返回格式：

```json
{
  "status": "success",
  "summary": "Found 3 result(s) for 'RAG'.",
  "data": { },
  "source_ids": ["chunk_1"],
  "next_actions": []
}
```

---

## Dashboard

```bash
uv run askbook dashboard                    # http://localhost:8501
uv run askbook dashboard --port 9000
```

| 页面 | 内容 |
|------|------|
| **1. 系统总览** | 今日 Query 数、P50 延迟、Token 总量、组件健康状态 |
| **2. 数据浏览** | Chroma Collection 列表、文档列表、Chunk 内容预览 |
| **3. Pipeline 监控** | Ingestion 记录列表、Plotly Gantt 甘特图、Query Rewrite 审计 |
| **4. Trace 查看器** | 按 trace_id 查看完整调用链路 |
| **5. 评估与质量** | 检索指标、Ragas + LLM-judge 评分、Harness 健康卡片、用户反馈 |

---

## 评估体系

### 检索指标（4 种）

`hit_rate`、`mrr`、`recall@k`、`ndcg@k`

### 答案质量（2 种）

**Ragas** `faithfulness`（忠实度）、`answer_relevancy`（相关性）
**LLM-judge** 并发判卷

### 使用

```bash
uv run askbook ingest examples/docs/seed --collection demo
uv run askbook eval --dataset datasets/seed_manual.yaml --collection demo --k 5
```

每次评估写入 `eval_runs/<timestamp>.json`。CI 有 golden baseline 门禁，指标下降超 0.05 会挂。

---

## Harness 健康指标

Dashboard 页面 5 实时呈现，颜色编码：

| 指标 | 目标 | 说明 |
|------|------|------|
| **completion_rate** | >= 95% | MCP 工具调用成功率 |
| **retries_per_task** | <= 1.2 | 每次成功调用的平均重试次数 |
| **pass@1** | >= 85% | 一次即成功的任务占比 |
| **cost_per_task** | <= ¥0.05 | 每次成功 ask 调用的平均成本 |

### 反模式 CI 检查

6 种反模式静态分析：context rot、hallucinated completion、model drift 等。

```bash
uv run python scripts/anti_pattern_check.py    # 6/6 PASS 即无违规
```

---

## 可观测性

每次 `ingest` / `query` / MCP 调用自动写入 JSONL trace 到 `~/.askbook/traces/YYYY-MM-DD.jsonl`：

- **PII 脱敏** — 手机号、邮箱、API Token 自动替换
- **自动清理** — 保留 7 天，滚动删除
- **Dashboard 可视化** — 甘特图、Query Rewrite Diff、HyDE 生成预览

---

## Fallback 链

主 LLM 不可用时自动降级：

```yaml
llm:
  provider: dashscope
  model: qwen-plus
  fallback_chain:
    - provider: openai
      model: gpt-4o-mini
    - provider: ollama
      model: qwen2.5:7b
```

`FallbackProvider` 按顺序尝试，全部失败抛出 `ProviderFallbackExhaustedError`。

---

## 开发

```bash
uv run pytest -q                                    # 全量测试 (300+)
uv run ruff check .                                 # Lint
uv run mypy --strict src/                           # 类型检查
uv run pytest --cov=src/askbook --cov-fail-under=80 # 覆盖率 >= 80%
uv run python scripts/anti_pattern_check.py         # 反模式扫描
```

---

## FAQ

**`uv: command not found`** → 重开终端，确认 `~/.local/bin` 在 PATH 中

**`ollama: command not found`** → 从开始菜单/启动台打开 Ollama；Linux 执行 `sudo systemctl start ollama`

**导入卡住** → 确认 ollama 在运行；大文件首次处理较慢

**回答质量差** → 确认文档覆盖问题域；换更大模型或启用百炼/OpenAI

**API Key 不生效** → 确认 `.env` 在项目根目录；或直接 `export` 设环境变量

**百炼返回空或超时** → 配置 `fallback_chain` 自动降级

**路径含空格/中文** → 加引号：`uv run askbook ingest "D:\我的文档\笔记.md" --collection notes`

**磁盘不够** → 数据在 `~/.askbook/`，改 `ASKBOOK_DATA_DIR` 指向大磁盘
