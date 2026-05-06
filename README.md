# askbook

## 我们在构建什么

askbook 是一个**本地优先的私有知识库 RAG 系统**。把 PDF、Word、Markdown、TXT 等文档导入后，通过自然语言提问即可获得带原文引用的准确回答——数据全程不出机器，无需上传任何云端。

同时以 **MCP Server**（Model Context Protocol）对外暴露，让 Claude Desktop、VS Code Copilot 等 AI 助手可以直接调用你的私有知识库，实现「AI + 你的文档」无缝协作。

## 面向谁

| 角色 | 使用场景 |
|------|---------|
| **个人研究者 / 学生** | 论文、笔记、技术文档堆积太多——用自然语言检索并问答，自动标注来源 |
| **本地优先用户 / 小团队** | 数据敏感、合规要求严、不能上传云端——在本地/内网服务器部署 |
| **AI 应用开发者** | 通过 MCP 协议给 Claude Desktop 等 AI 助手挂载私有知识库 |
| **技术面试评估** | 展示 RAG + MCP 全栈工程能力：混合检索、多模态摄入、全链路 Trace、Harness 质量门禁 |

## 架构速览

```
你的文档 (PDF/MD/TXT/Word)
    │
    ▼
Ingestion Pipeline ── chunk → embed → ChromaDB + BM25
    │
    ▼
Query Pipeline ── hybrid retrieval (dense+sparse+RRF) → rerank → LLM 合成
    │
    ├── CLI:   uv run askbook query ask "xxx" --collection demo
    ├── MCP:   uv run askbook serve --collection demo  (Claude Desktop 直接调)
    └── Web:   uv run askbook dashboard  (Streamlit 5 页监控面板)
```

## 快速开始

**前置条件：** Python 3.12+、8GB+ 内存、10GB+ 磁盘

### 模式 A — 纯本地（零云端依赖，零 API 成本）

**1. 安装 Ollama（本地 LLM 运行时）**

- Windows / macOS：https://ollama.com/download 下载安装
- Linux：`curl -fsSL https://ollama.com/install.sh | sh`
- 启动后会在 `localhost:11434` 监听

**2. 拉取模型**

```bash
ollama pull qwen2.5:7b          # 对话模型 (~4.5G)，中文场景推荐
ollama pull nomic-embed-text    # embedding 模型，文档向量化用
```

> 模型选择：显存 >= 16G 可选 `qwen2.5:14b`；纯英文场景选 `llama3.1:8b`。

**3. 安装 uv（Python 包管理器）**

```bash
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**4. 克隆项目并安装**

```bash
git clone https://github.com/heylong7/askbook.git
cd askbook
uv sync --all-extras
cp .env.example .env            # 纯本地模式无需修改，API Key 留空即可
```

**5. 导入文档**

```bash
uv run askbook ingest examples/docs/seed --collection demo    # 先试用示例数据
uv run askbook ingest /path/to/docs --collection my-docs       # 你自己的文档
uv run askbook ingest /path/to/single.pdf --collection my-docs
```

**6. 提问**

```bash
# 问答模式：检索 + LLM 合成答案 + 引用来源
uv run askbook query ask "如何配置 Ollama？" --collection demo

# 搜索模式：只返回相关 chunk，不调 LLM
uv run askbook query search "混合检索原理" --collection demo
```

### 模式 B — 云端加速（阿里云百炼 API）

本地模型（7B/14B）在复杂问答场景下可能不如云端大模型。百炼（阿里云大模型平台）按量计费，qwen-plus 约 ¥0.004/千 token。

**1. 获取 API Key**

- 注册：https://bailian.console.aliyun.com
- 开通「模型服务」→ 创建 API Key → 复制 `sk-xxxxxxxx`

**2. 配置环境变量**

编辑项目根目录的 `.env`：

```bash
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

> 纯本地模式不需要填 API Key。需要时才填。

**3. 用百炼配置启动**

项目提供 `configs/ollama-only.yaml`（纯本地）。创建 `configs/bailian.yaml`：

```yaml
# 百炼 + 本地 embedding
llm:
  provider: dashscope
  model: qwen-plus
  temperature: 0.1

embedding:
  provider: bge-m3           # embedding 仍走本地 bge-m3，零成本

query:
  top_k: 10
  enable_rewrite: true
```

```bash
uv run askbook serve --collection demo --config configs/bailian.yaml
```

**4. 可选：配置 Fallback 链**

```yaml
llm:
  provider: dashscope
  model: qwen-plus
  fallback_chain:
    - provider: ollama
      model: qwen2.5:7b
```

百炼超时或不可用时，自动降级到本地 Ollama。

## 配置详解

### 配置优先级

```
环境变量 (ASKBOOK_*)  >  .env 文件  >  --config YAML  >  defaults.yaml（内置）
```

环境变量用双下划线 `__` 表示嵌套层级：

```bash
export ASKBOOK_LLM__PROVIDER=dashscope
export ASKBOOK_LLM__MODEL=qwen-plus
export ASKBOOK_LLM__TEMPERATURE=0.0
```

### Provider 选项

| 组件 | Provider | 需要 API Key | 说明 |
|------|----------|-------------|------|
| **LLM** | `ollama` | 否 | 本地运行，需先 `ollama pull` 拉模型 |
| **LLM** | `dashscope` | 是 (`DASHSCOPE_API_KEY`) | 阿里云百炼，qwen 系列模型 |
| **LLM** | `stub` | 否 | 测试用，返回固定回答 |
| **Embedding** | `bge-m3` | 否 | 本地运行 BAAI/bge-m3，1024 维 |
| **Embedding** | `stub` | 否 | 测试用，哈希假向量 |
| **Vector Store** | `chroma` | 否 | 本地 ChromaDB，数据存 `~/.askbook/chroma` |
| **Reranker** | `bge-v2-m3` | 否 | 本地 Cross-Encoder 精排 |
| **Reranker** | `stub` | 否 | 测试用 |

### 完整配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm.provider` | `ollama` | LLM provider |
| `llm.model` | `qwen2.5:7b` | 模型名称 |
| `llm.temperature` | `0.1` | 生成温度 |
| `llm.max_tokens` | `4096` | 单次最大输出 token 数 |
| `llm.fallback_chain` | `[]` | Fallback 链，如 `[{provider: ollama, model: qwen2.5:7b}]` |
| `llm.token_limit_per_call` | `8000` | 单次调用输入 token 上限 |
| `llm.quota_per_hour` | `100` | 每小时最大调用次数 |
| `embedding.provider` | `bge-m3` | Embedding provider |
| `embedding.model` | `BAAI/bge-m3` | Embedding 模型名 |
| `embedding.device` | `auto` | 推理设备（`cpu` / `cuda` / `auto`） |
| `embedding.batch_size` | `32` | Embedding 批处理大小 |
| `vectorstore.path` | `~/.askbook/chroma` | ChromaDB 存储路径 |
| `ingestion.chunk_size` | `600` | Chunk 大小（字符） |
| `ingestion.chunk_overlap` | `80` | Chunk 重叠（字符） |
| `ingestion.embed_concurrency` | `4` | Embedding 并发数 |
| `query.top_k` | `10` | 检索返回数 |
| `query.rerank_top_k` | `5` | 重排序后保留数 |
| `query.rrf_k` | `60` | RRF 融合参数 |
| `query.enable_rewrite` | `true` | 启用查询改写 |
| `query.enable_hyde` | `false` | 启用 HyDE 假设文档 |
| `observability.trace_dir` | `~/.askbook/traces` | Trace 日志目录 |
| `observability.retention_days` | `7` | Trace 保留天数 |
| `observability.pii_redaction` | `true` | PII 脱敏（手机号/邮箱/Token） |
| `data_dir` | `~/.askbook` | 数据根目录 |

> 环境变量格式：`ASKBOOK_LLM__PROVIDER=dashscope`（双下划线分隔嵌套 key）

## MCP Server（接入 Claude Desktop）

```bash
uv run askbook serve --collection demo
```

将 `examples/claude_desktop_mcp.json` 合并到 Claude Desktop 的 MCP 配置中：

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

重启 Claude Desktop 后可用 **6 个 MCP 工具**：

| 工具 | 类型 | 说明 |
|------|------|------|
| `search` | 核心 | 语义搜索，返回排序片段 + 来源 |
| `ask` | 核心 | 完整 RAG 问答，答案 + 引用 |
| `list_collections` | 核心 | 列出所有 collection 及统计 |
| `get_document_summary` | 核心 | 指定文档摘要信息 |
| `trace_lookup` | 诊断 | 按 trace_id 查看链路事件 |
| `collection_stats` | 诊断 | Collection 详细统计（chunk 数 / 文档数 / 磁盘大小） |

**Tool 返回格式：**

```json
{
  "status": "success",
  "summary": "Found 3 result(s) for 'RAG'.",
  "data": { },
  "source_ids": ["chunk_1"],
  "next_actions": []
}
```

## Dashboard

```bash
uv run askbook dashboard              # http://localhost:8501
uv run askbook dashboard --port 9000  # 自定义端口
```

5 个页面：

| 页面 | 功能 |
|------|------|
| 系统总览 | 今日 Query 数 / P50 延迟 / Token 总量 / 组件健康 |
| 数据浏览 | Chroma Collection 文档列表 + Chunk 预览 |
| Pipeline 监控 | Ingestion 记录 + Plotly Gantt 甘特图 |
| Trace 查看器 | Query Rewrite Diff、HyDE 生成文档、Trace 详情 |
| 评估与质量 | 检索指标（Hit Rate / MRR / Recall / NDCG）、Ragas + LLM-judge 评分、Harness 健康卡片、用户反馈 |

## 评估

内置 20 条人工标注 QA 数据集，6 种评估指标：

**检索指标（4 种）：** `hit_rate`、`mrr`、`recall@k`、`ndcg@k`

**答案质量（2 种）：** Ragas `faithfulness`（忠实度）、`answer_relevancy`（相关性）+ LLM-judge 并发判卷

```bash
uv run askbook ingest examples/docs/seed --collection demo
uv run askbook eval --dataset datasets/seed_manual.yaml --collection demo --k 5
```

每次评估写入 `eval_runs/<timestamp>.json`。CI 有 golden baseline 门禁，指标下降超 0.05 会挂。

## Harness 健康指标

askbook 内置 Harness 健康监控，在 Dashboard 页面 5 实时呈现，颜色编码：

| 指标 | 目标 | 说明 |
|------|------|------|
| **completion_rate** | >= 95% | MCP 工具调用成功率 |
| **retries_per_task** | <= 1.2 | 每次成功调用的平均重试次数 |
| **pass@1** | >= 85% | 一次即成功的任务占比 |
| **cost_per_task** | <= ¥0.05 | 每次成功 ask 调用的平均成本 |

**Anti-pattern CI 检查**：每次 CI 对 6 种反模式做静态分析（context rot、hallucinated completion、model drift 等），确保代码提交符合质量门禁。详见 `DEV_SPEC.md` §30.3。

```bash
uv run python scripts/anti_pattern_check.py    # 6/6 PASS 表示无违规
```

## 开发

```bash
uv run pytest -q                          # 全量测试
uv run ruff check .                       # Lint
uv run mypy --strict src/                 # 类型检查
uv run pytest --cov=src/askbook --cov-fail-under=80 -q  # 覆盖率 >= 80%
uv run python scripts/anti_pattern_check.py              # Harness 反模式扫描
```

379 个测试，覆盖率 >= 80%。详细规范见 `DEV_SPEC.md`。

## 可观测性

每次 `ingest` / `query` / MCP 调用自动写入 JSONL trace 到 `~/.askbook/traces/YYYY-MM-DD.jsonl`。支持 PII 脱敏（手机号/邮箱/Token），保留 7 天。

## FAQ

**`uv: command not found`** → 重开终端，确认 `~/.local/bin` 在 PATH 里

**`ollama: command not found`** → Windows/macOS 从开始菜单/启动台打开 Ollama；Linux 执行 `sudo systemctl start ollama`

**导入卡住** → 确认已 `ollama pull nomic-embed-text`；大文件首次处理慢

**回答质量差** → 确认文档覆盖了问题域；GPU 机器换 `qwen2.5:14b` 或启用百炼

**`DASHSCOPE_API_KEY` 不生效** → 确认 `.env` 在项目根目录；或 `export DASHSCOPE_API_KEY=sk-xxx` 直接设环境变量

**百炼返回空或超时** → 配置 `fallback_chain` 自动降级到 Ollama

**路径含空格/中文** → 必须加引号：`uv run askbook ingest "D:\我的文档\笔记.md" --collection notes`

**磁盘不够** → 数据在 `~/.askbook/`，改 `ASKBOOK_DATA_DIR` 指向大磁盘
