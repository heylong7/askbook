# 配置体系

askbook 采用三层配置策略：YAML 文件定义结构化设置、环境变量注入密钥、pydantic 模型在启动时校验。

## 三层架构

1. **YAML 配置文件**：定义所有非敏感设置。支持多份 YAML 适配不同环境（本地 Ollama vs 云端百炼）
2. **环境变量**：注入 API Key 等密钥，使用 `ASKBOOK_` 前缀，双下划线表示嵌套
3. **pydantic 校验**：启动时对所有配置做严格类型检查，无效配置立即报错并指明字段

## 环境变量覆盖

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASKBOOK_DATA_DIR` | `~/.askbook` | 数据和 BM25 索引目录 |
| `ASKBOOK_LLM__PROVIDER` | `ollama` | LLM 后端 |
| `ASKBOOK_LLM__MODEL` | `qwen2.5:7b` | LLM 模型名 |
| `ASKBOOK_EMBEDDING__PROVIDER` | `bge-m3` | Embedding 后端 |
| `ASKBOOK_VECTORSTORE__PATH` | `~/.askbook/chroma` | ChromaDB 存储路径 |

## 配置域

主配置文件覆盖以下域：

- **llm**：provider、model、temperature、max_tokens、fallback_chain、quota_per_hour
- **embedding**：provider、model、normalize、device、batch_size
- **vectorstore**：provider（chroma）、持久化路径
- **ingestion**：chunk_size、chunk_overlap、embed_concurrency、vision_concurrency、chroma_concurrency、enrich_llm（可选）
- **query**：top_k、rerank_top_k、rrf_k、enable_rewrite、enable_hyde、enable_llm_rerank
- **mcp**：transport、system_prompt_max_tokens
- **observability**：trace_dir、flush_interval_seconds、pii_redaction、enabled、retention_days、dashboard_port

## LLM 后端

| Provider | 说明 | 所需环境变量 |
|----------|------|-------------|
| `ollama` | 本地 Ollama | 无 |
| `dashscope` | 阿里云百炼 | `DASHSCOPE_API_KEY` |
| `openai` | OpenAI | `OPENAI_API_KEY` |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` |
| `stub` | 测试桩 | 无 |

## Embedding 后端

| Provider | 说明 | 所需环境变量 |
|----------|------|-------------|
| `bge-m3` | 本地 BGE-M3 | 无 |
| `openai` | OpenAI Embeddings | `OPENAI_API_KEY` |
| `dashscope` | 百炼 TextEmbedding | `DASHSCOPE_API_KEY` |
| `stub` | 测试桩 | 无 |

## Fallback 链

LLM 配置支持 fallback_chain（数组），当主 provider 超时或出错时自动切换到下一个。例如：主用百炼，fallback 到本地 Ollama。
