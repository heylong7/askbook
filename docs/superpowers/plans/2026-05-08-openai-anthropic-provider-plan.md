# OpenAI / Anthropic LLM + Embedding Provider 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 OpenAI + Anthropic LLM provider 和 OpenAI + DashScope embedding provider，复用现有 schema。

**Architecture:** 4 个新文件（2 LLM provider + 2 embedder），全部实现已有 Protocol/ABC，通过 `ServiceRegistry.build_llm()` / `build_embedder()` 注册，密钥从环境变量读取。

**Tech Stack:** httpx（LLM）、dashscope SDK（DashScope embedder）、pytest + respx（测试）

---

### Task 1: OpenAI LLM Provider

**Files:**
- Create: `src/askbook/providers/openai_provider.py`
- Create: `tests/unit/test_openai_provider.py`

- [ ] **Step 1: 写测试 — provider 基本属性**

```python
# tests/unit/test_openai_provider.py

def test_provider_name():
    """provider_name 返回 'openai'"""
    ...

def test_conforms_to_protocol():
    """OpenAIProvider 满足 LLMProviderProtocol"""
    ...

def test_reads_api_key_from_env(monkeypatch):
    """从 OPENAI_API_KEY 环境变量读取密钥"""
    ...
```

- [ ] **Step 2: 跑测试确认 FAIL**

```bash
pytest tests/unit/test_openai_provider.py -v
```

- [ ] **Step 3: 实现 OpenAIProvider 骨架**

```python
# src/askbook/providers/openai_provider.py
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, *, model="gpt-4o", temperature=0.1, max_tokens=4096, api_key=None):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]

    @property
    def provider_name(self) -> str:
        return "openai"

    def complete(self, prompt, *, temperature=None, max_tokens=None) -> LLMResponse:
        return asyncio.run(self.acomplete(prompt, temperature=temperature, max_tokens=max_tokens))
```

- [ ] **Step 4: 写测试 — complete 调用成功**

```python
@respx.mock
def test_complete_returns_llm_response():
    """mock OpenAI API 返回正常响应，验证 LLMResponse 各字段"""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o"
        })
    )
    # 断言: content, provider, model, usage 正确
    ...

@respx.mock
def test_complete_raises_on_timeout():
    """超时时抛出 ProviderTimeoutError"""
    ...

@respx.mock
def test_complete_raises_on_unauthorized():
    """401 时抛出 ProviderError"""
    ...
```

- [ ] **Step 5: 跑测试确认 FAIL**

- [ ] **Step 6: 实现 complete / acomplete / astream**

关键实现片段：

```python
async def _do_acomplete(self, prompt, *, temperature, max_tokens):
    eff_temp = temperature if temperature is not None else self._temperature
    eff_tokens = max_tokens if max_tokens is not None else self._max_tokens
    body = {
        "model": self._model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": eff_temp,
        "max_tokens": eff_tokens,
    }
    headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=self._timeout) as client:
        resp = await client.post(self._endpoint, json=body, headers=headers)
        resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = TokenUsage(
        prompt_tokens=data["usage"]["prompt_tokens"],
        completion_tokens=data["usage"]["completion_tokens"],
        total_tokens=data["usage"]["total_tokens"],
    )
    return LLMResponse(content=content, usage=usage, model=data.get("model", self._model), provider="openai", latency_ms=...)
```

- [ ] **Step 7: 跑测试确认全部 PASS**

```bash
pytest tests/unit/test_openai_provider.py -v
```

- [ ] **Step 8: Commit**

```bash
git add src/askbook/providers/openai_provider.py tests/unit/test_openai_provider.py
git commit -m "feat: add OpenAI LLM provider"
```

---

### Task 2: Anthropic LLM Provider

**Files:**
- Create: `src/askbook/providers/anthropic_provider.py`
- Create: `tests/unit/test_anthropic_provider.py`

- [ ] **Step 1: 写测试 — provider 基本属性**

```python
# tests/unit/test_anthropic_provider.py

def test_provider_name():
    """provider_name 返回 'anthropic'"""
    ...

def test_conforms_to_protocol():
    """AnthropicProvider 满足 LLMProviderProtocol"""
    ...

def test_reads_api_key_from_env(monkeypatch):
    """从 ANTHROPIC_API_KEY 环境变量读取密钥"""
    ...
```

- [ ] **Step 2: 跑测试确认 FAIL**

- [ ] **Step 3: 实现 AnthropicProvider 骨架**

```python
class AnthropicProvider(BaseLLMProvider):
    def __init__(self, *, model="claude-sonnet-4-6", temperature=0.1, max_tokens=4096, api_key=None):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    @property
    def provider_name(self) -> str:
        return "anthropic"
```

- [ ] **Step 4: 写测试 — complete 调用成功**

```python
@respx.mock
def test_complete_returns_llm_response():
    """mock Anthropic Messages API 返回正常响应"""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-4-6"
        })
    )
    # 断言: content, provider, model, usage 正确
    ...
```

- [ ] **Step 5: 跑测试确认 FAIL**

- [ ] **Step 6: 实现 complete / acomplete / astream**

关键差异：Anthropic API 请求头包含 `x-api-key` 和 `anthropic-version: 2023-06-01`，body 中 `messages` 的 content 可以是纯字符串。响应中的 token 字段名是 `input_tokens`/`output_tokens`（不是 prompt/completion）。

- [ ] **Step 7: 跑测试确认全部 PASS**

- [ ] **Step 8: Commit**

```bash
git add src/askbook/providers/anthropic_provider.py tests/unit/test_anthropic_provider.py
git commit -m "feat: add Anthropic LLM provider"
```

---

### Task 3: OpenAI Embedder

**Files:**
- Create: `src/askbook/embeddings/openai_embedder.py`
- Create: `tests/unit/test_openai_embedder.py`

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_openai_embedder.py

@respx.mock
def test_embed_query_returns_vector():
    """embed_query 返回 float 列表，维度正确"""
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small"
        })
    )
    # assert isinstance(result, list), all floats

@respx.mock
def test_embed_batch():
    """embed_batch 批量返回多个向量"""

def test_conforms_to_protocol():
    """OpenAIEmbedder 满足 EmbedderProtocol"""

def test_dimension_from_api_response():
    """首次 API 调用后 dimension 属性可用"""
```

- [ ] **Step 2: 跑测试确认 FAIL**

- [ ] **Step 3: 实现 OpenAIEmbedder**

```python
class OpenAIEmbedder:
    def __init__(self, *, model="text-embedding-3-small", api_key=None):
        self.model_name = model
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._endpoint = "https://api.openai.com/v1/embeddings"
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("dimension not available — call embed_* first")
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def embed_passage(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return self._call_api(texts)
```

API 调用：`POST` 到 `/v1/embeddings`，body `{"model": ..., "input": texts}`，从 `response["data"][i]["embedding"]` 取向量。

- [ ] **Step 4: 跑测试确认全部 PASS**

- [ ] **Step 5: Commit**

```bash
git add src/askbook/embeddings/openai_embedder.py tests/unit/test_openai_embedder.py
git commit -m "feat: add OpenAI embedding provider"
```

---

### Task 4: DashScope Embedder

**Files:**
- Create: `src/askbook/embeddings/dashscope_embedder.py`
- Create: `tests/unit/test_dashscope_embedder.py`

- [ ] **Step 1: 写测试**

```python
# tests/unit/test_dashscope_embedder.py

def test_reads_api_key_from_env(monkeypatch):
    """从 DASHSCOPE_API_KEY 环境变量读取"""

def test_embed_query_returns_vector():
    """mock dashscope.TextEmbedding.call，验证返回值格式"""
    # 用 patch("dashscope.TextEmbedding") mock 调用
    ...

def test_conforms_to_protocol():
    """DashScopeEmbedder 满足 EmbedderProtocol"""
```

- [ ] **Step 2: 跑测试确认 FAIL**

- [ ] **Step 3: 实现 DashScopeEmbedder**

参考现有 `DashScopeQwenProvider` 的 dashscope SDK 用法模式，调 `dashscope.TextEmbedding.call(model=..., input=...)`。

```python
class DashScopeEmbedder:
    def __init__(self, *, model="text-embedding-v3", api_key=None):
        self.model_name = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        try:
            import dashscope
            dashscope.api_key = self._api_key
        except ImportError:
            pass

    @property
    def dimension(self) -> int:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_passage(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        ...
```

- [ ] **Step 4: 跑测试确认全部 PASS**

- [ ] **Step 5: Commit**

```bash
git add src/askbook/embeddings/dashscope_embedder.py tests/unit/test_dashscope_embedder.py
git commit -m "feat: add DashScope embedding provider"
```

---

### Task 5: Registry 注册 + 集成验证

**Files:**
- Modify: `src/askbook/core/registry.py:41-63` — `build_llm()` 加分支
- Modify: `src/askbook/core/registry.py:83-100` — `build_embedder()` 加分支
- Modify: `src/askbook/providers/__init__.py`
- Modify: `src/askbook/embeddings/__init__.py`

- [ ] **Step 1: 更新 build_llm()**

在 `build_llm()` 中 `if provider == "dashscope":` 之后加两个分支：

```python
if provider == "openai":
    from askbook.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(
        model=config.model, temperature=config.temperature, max_tokens=config.max_tokens
    )
if provider == "anthropic":
    from askbook.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(
        model=config.model, temperature=config.temperature, max_tokens=config.max_tokens
    )
```

同时更新错误消息中的 Supported 列表，加入 `'openai'`, `'anthropic'`。

- [ ] **Step 2: 更新 build_embedder()**

在 `build_embedder()` 中 `if provider in ("bge-m3", "bge_m3"):` 之后加：

```python
if provider == "openai":
    from askbook.embeddings.openai_embedder import OpenAIEmbedder

    return OpenAIEmbedder(model=config.model)
if provider == "dashscope":
    from askbook.embeddings.dashscope_embedder import DashScopeEmbedder

    return DashScopeEmbedder(model=config.model)
```

同时更新错误消息中的 Supported 列表，加入 `'openai'`, `'dashscope'`。

- [ ] **Step 3: 更新 __init__.py**

`src/askbook/providers/__init__.py` 和 `src/askbook/embeddings/__init__.py` 加 re-export（按现有空文件风格，或仅补充 `__all__`）。

- [ ] **Step 4: 跑全部测试**

```bash
pytest tests/ -v --cov=src/askbook/providers --cov=src/askbook/embeddings --cov=src/askbook/core/registry.py -k "not (bge or chroma or dashboard or eval)"
```

- [ ] **Step 5: 手动冒烟测试**

创建临时 YAML 测试配置切换：

```yaml
# 测试 OpenAI LLM
llm:
  provider: openai
  model: gpt-4o
```

```bash
export OPENAI_API_KEY="sk-xxx"
askbook --config /tmp/test-openai.yaml ask "test"
```

- [ ] **Step 6: Commit**

```bash
git add src/askbook/core/registry.py src/askbook/providers/__init__.py src/askbook/embeddings/__init__.py
git commit -m "feat: register OpenAI/Anthropic LLM and OpenAI/DashScope embedding providers"
```
