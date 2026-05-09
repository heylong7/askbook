# OpenAI / Anthropic LLM + Embedding Provider 接入设计

## 目标

新增 4 个 provider：
- LLM：`OpenAIProvider`（Chat Completions）、`AnthropicProvider`（Messages）
- Embedding：`OpenAIEmbedder`（Embeddings API）、`DashScopeEmbedder`（百炼 TextEmbedding）

全部复用现有 `LLMConfig` / `EmbeddingConfig` schema，零 schema 变更。

## 设计原则

- **零 schema 变更**：`LLMConfig` 和 `EmbeddingConfig` 不变，仅扩展 `provider` 可选值
- **密钥走环境变量**：`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DASHSCOPE_API_KEY`
- **官方默认地址**：不暴露 `base_url`
- **延续现有模式**：与 `ollama`/`dashscope`/`bge-m3`/`stub` 同层、同接口

## LLM 配置示例

```yaml
# OpenAI
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.1
  max_tokens: 4096

# Anthropic
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  temperature: 0.1
  max_tokens: 4096
```

fallback_chain、token_limit_per_call、quota_per_hour 等现有字段继续可用。

## Embedding 配置示例

```yaml
# OpenAI embedding
embedding:
  provider: openai
  model: text-embedding-3-small
  batch_size: 32

# 百炼 embedding
embedding:
  provider: dashscope
  model: text-embedding-v3
  batch_size: 32

# 本地 bge-m3（现有，不变）
embedding:
  provider: bge-m3
  model: BAAI/bge-m3
  normalize: true
  device: auto
```

`normalize`、`device` 仅 bge-m3 使用，API 类 embedder 忽略。

## 改动范围

### 新建文件

**`src/askbook/providers/openai_provider.py`**
- `OpenAIProvider(BaseLLMProvider)`，调 `POST https://api.openai.com/v1/chat/completions`
- 用 `httpx` 发请求，单条 user message 包装
- 密钥 `os.environ["OPENAI_API_KEY"]`

**`src/askbook/providers/anthropic_provider.py`**
- `AnthropicProvider(BaseLLMProvider)`，调 `POST https://api.anthropic.com/v1/messages`
- 请求头 `x-api-key` + `anthropic-version: 2023-06-01`
- 密钥 `os.environ["ANTHROPIC_API_KEY"]`

**`src/askbook/embeddings/openai_embedder.py`**
- `OpenAIEmbedder`，实现 `EmbedderProtocol`
- 调 `POST https://api.openai.com/v1/embeddings`
- `dimension` 从首次 API 响应中获取

**`src/askbook/embeddings/dashscope_embedder.py`**
- `DashScopeEmbedder`，实现 `EmbedderProtocol`
- 调百炼 TextEmbedding API，复用现有 dashscope SDK
- `dimension` 从 API 响应获取

### 修改文件

**`src/askbook/core/registry.py`**
- `build_llm()` 加 `openai` / `anthropic` 分支
- `build_embedder()` 加 `openai` / `dashscope` 分支

**`src/askbook/providers/__init__.py`** — 新 provider re-export

**`src/askbook/embeddings/__init__.py`** — 新 embedder re-export

### 不修改

`schema.py`、`settings.py`、`defaults.yaml`、`base.py`、所有 pipeline/query 文件。

## 不在此范围

- 可视化配置切换 / 分块预览
- 自定义 API 地址
- 密钥写入 YAML
