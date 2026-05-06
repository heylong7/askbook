# 面试题库 — askbook RAG 系统

## 目录

1. [RAG 基础](#1-rag-基础)
2. [MCP 协议](#2-mcp-协议)
3. [质量评估](#3-质量评估)
4. [Harness 工程](#4-harness-工程)
5. [Python 工程](#5-python-工程)

---

## 1. RAG 基础

### 1.1 什么是 RAG？它是如何工作的？

RAG（Retrieval-Augmented Generation）是一种将检索与生成相结合的架构。工作流程分为三步：(1) 将用户 query 编码为向量，从向量数据库中检索最相关的文档片段；(2) 将检索到的上下文与原始 query 拼接成 prompt；(3) 将 prompt 送入 LLM 生成带引用的答案。RAG 的核心优势在于无需微调即可引入外部知识，并有效缓解 LLM 的幻觉问题。

### 1.2 向量数据库与传统数据库有什么区别？

传统数据库（如 MySQL）基于精确匹配和范围查询，适合结构化数据；向量数据库（如 ChromaDB、Milvus）基于向量相似度搜索（余弦相似度、欧氏距离等），适合非结构化的语义匹配。RAG 中通常两者并用：向量数据库做语义检索，SQLite 等传统数据库存储文档元数据和 ingestion 记录。

### 1.3 Chunking 策略有哪些？各自的权衡是什么？

常见策略：固定大小分块（按 token/字符数切分）、递归分块（按段落/句子层级递归拆分至目标大小）、语义分块（基于 embedding 相似度检测边界）、基于文档结构分块（按 Markdown heading、HTML section）。权衡在于：chunk 太小会丢失上下文连贯性导致检索噪声，chunk 太大会引入无关信息并降低检索精度，实践中通常 400-800 token 配合 10-20% overlap 作为起点。

### 1.4 BM25 与稠密检索（Dense Retrieval）的区别是什么？

BM25 是基于词频-逆文档频率（TF-IDF）的稀疏检索方法，依赖关键词精确命中，无需训练，适合短文本和术语密集的场景。稠密检索使用神经网络将 query 和文档映射到稠密向量空间，能捕捉同义词和语义关联，但对训练数据和推理资源有要求。askbook 使用 BM25 与稠密检索的混合方案（RRF 融合排序），兼顾召回率与精确率。

### 1.5 Embedding 在 RAG 中扮演什么角色？

Embedding 是将文本映射为固定维度稠密向量的过程，是 RAG 检索阶段的核心。一个好的 embedding 模型应使语义相近的文本在向量空间中距离相近。常见的选项包括 BGE-M3（askbook 默认）、text-embedding-ada-002 等。选择 embedding 模型时需考虑维度（影响存储和检索速度）、语言支持（是否需要中英双语）和归一化方式。

---

## 2. MCP 协议

### 2.1 什么是 MCP？它与 REST API 有何不同？

MCP（Model Context Protocol）是 Anthropic 提出的、专为 LLM 应用设计的工具调用协议，以 JSON-RPC 2.0 为基础。与 REST 不同，MCP 不需要 HTTP 路由、状态码和复杂的鉴权——它通过 stdio 或 HTTP SSE 传输请求-响应消息，每个工具是一个函数签名（输入 schema + 处理器 + 响应合约），非常适合 Claude Desktop 等 AI 客户端直接调用。

### 2.2 MCP 工具定义包含哪些要素？

一个 MCP 工具包含三个部分：(1) 输入 Schema —— 使用 Pydantic BaseModel 定义参数类型和约束；(2) Handler —— 纯函数处理器，接收输入和依赖注入（ServerDeps），返回 ToolResponse；(3) 注册 —— 工具名与 (input_model, handler) 的映射，MCP Server 启动时自动构建路由表。askbook 的 `TOOL_REGISTRY` 字典就是这种模式的典型实现。

```python
TOOL_REGISTRY: dict[str, tuple[type[BaseModel], Callable]] = {
    "search": (SearchInput, handle_search),
    "ask": (AskInput, handle_ask),
}
```

### 2.3 MCP Server 的实现模式是什么？

MCP Server 采用三层架构：(1) contracts 层 —— Pydantic I/O 模型；(2) tools 层 —— 纯函数 handler，通过 `ServerDeps` 显式接收依赖；(3) server 层 —— MCP SDK 包装，启动 transport 并注册工具。Handler 不直接操作 IO，所有副作用（数据库查询、Trace 写入）通过依赖接口完成，便于单元测试。

### 2.4 Claude Desktop 如何与 MCP Server 集成？

用户在 Claude Desktop 的 MCP 配置文件中注册 server 名称、启动命令（command + args）和环境变量。Claude Desktop 以子进程方式启动 MCP Server，通过 stdio 交换 JSON-RPC 消息。用户在对话中自然语言触发的工具调用，会自动路由到对应的 MCP handler 执行并返回结果。

### 2.5 MCP Transport：stdio 与 HTTP 的区别是什么？

stdio transport 以子进程方式运行，通过标准输入输出通信，适合本地开发（无需网络配置），但每个工具调用共享同一进程生命周期。HTTP transport（SSE）基于 Server-Sent Events，支持远程调用和独立部署，但需要处理端口冲突和鉴权。askbook 默认使用 stdio，支持通过配置切换到 HTTP SSE。

---

## 3. 质量评估

### 3.1 Hit Rate、MRR、NDCG 分别衡量什么？

Hit Rate 衡量 top-k 结果中是否包含至少一个相关文档（二值指标），简单直观但粒度粗糙。MRR（Mean Reciprocal Rank）关注第一个相关文档的排名位置，适合"只需一个正确答案"的场景。NDCG（Normalized Discounted Cumulative Gain）考虑多级相关性排序，对 top-k 中所有相关文档的位置做折扣加权，是三个指标中最敏感的。实践中三者常一起使用形成完整视图。

### 3.2 Recall@k 与 Precision@k 的区别是什么？

Recall@k = 检索到的相关文档数 / 总相关文档数，衡量"查全率"。Precision@k = 检索到的相关文档数 / 返回总数 k，衡量"查准率"。两者存在此消彼长的关系：增大 k 会提高 Recall 但可能降低 Precision。在 RAG 场景中 Recall@k 通常更关键，因为 LLM 可以容忍少量噪声但无法补偿缺失的信息。

### 3.3 Faithfulness 与 Answer Relevancy（Ragas）是什么意思？

Faithfulness 衡量答案中的陈述是否都能从检索上下文中找到依据，即"是否忠实于原文"——如果答案编造了上下文不包含的信息，faithfulness 会降低。Answer Relevancy 衡量答案与问题的相关程度，即"是否答非所问"——如果答案包含了与问题无关的信息，relevancy 会降低。askbook 集成 ragas 库计算这两个指标，在 Dashboard 页面 5 展示。

### 3.4 LLM-as-Judge 评估模式是什么？

LLM-as-Judge 使用一个 LLM（通常是一个更强的模型）作为评判者，对 QA 对打分。askbook 的实现使用 asyncio.Semaphore 控制并发度，对每个 QA 对分别进行 faithfulness 和 relevancy 评分，最后将 1-5 分的原始评分归一化为 0-1 范围。这种模式的优点是不需要标注数据，缺点是 judge 模型本身可能有偏见。

```python
JUDGE_FAITHFULNESS_PROMPT = """\
Rate the faithfulness of this answer to the provided context on a scale of 1-5.
Context: {context}
Question: {question}
Answer: {answer}
Score (1-5):"""
```

### 3.5 Golden Dataset 和基线回归测试是什么？

Golden Dataset 是人工标注的 QA 数据集，代表系统期望的正确行为。askbook 维护一个 20 条 seed_manual.yaml。运行时将当前检索结果与 golden 答案对比计算指标，并与已记录的基线分数（`v0.1_scores.json`）比较。CI 中 `pytest -m golden` 检测任何指标下降超过 0.05 即失败，防止无意的回归。

---

## 4. Harness 工程

### 4.1 什么是 Harness 工程？为什么它对 RAG 系统重要？

Harness 工程是 RAG 系统的质量保障层，涵盖监控、评估、成本控制和反模式检测。RAG 系统涉及多个组件（embedding、检索、重排序、LLM 生成），每个环节都可能引入质量退化——Harness 通过定义明确的 SLO（服务等级目标）和 CI 门禁来确保系统的可观测性和可维护性。

### 4.2 Completion Rate 监控什么？

Completion Rate 衡量 MCP 工具调用成功率（status=success 的调用数 / 总调用数），目标 >= 95%。它反映系统整体可用性——低于阈值可能表示 embedding 服务超时、ChromaDB 连接失败或 LLM 响应异常。askbook 从 JSONL Trace 中实时统计此指标并在 Dashboard 页面 5 用颜色编码呈现（达标绿色，超标红色）。

### 4.3 常见的 RAG 反模式有哪些？

askbook 的 Harness 层检测 6 种反模式（DEV_SPEC §30.3）：(1) Context Rot —— 检索上下文过时未更新；(2) Hallucinated Completion —— 答案包含上下文没有的信息；(3) Model Drift —— LLM 升级后行为不一致；(4) Empty Citation —— 成功响应但没有 source_id；(5) Zero-Recall Query —— 检索结果为空但 LLM 强行回答；(6) Token Overflow —— prompt 超过上下文窗口。每种反模式都在 CI 中做静态分析。

### 4.4 静态分析如何作为 CI 门禁？

静态分析在 CI 中检查 (1) Trace Schema 合规性（必填字段是否存在）；(2) ToolResponse 格式（source_ids 在 success 时必须非空）；(3) Baseline 回归（golden 指标下降超过阈值）。askbook 的 CI 门禁命令为 `ruff check && mypy --strict src/ && pytest --cov-fail-under=80`，确保代码质量符合 Harness 30.x 规范。

### 4.5 成本追踪与 Token 预算控制如何实现？

askbook 在每次 LLM 调用时估算 token 消耗并折算为人民币成本（基于模型 pricing），写入 Trace 的 `estimated_cost_cny` 标签。Dashboard 汇总每个任务的 Cost/Task 指标，阈值 <= ¥0.05。通过 `quota_per_hour` 和 `token_limit_per_call` 配置项做运行时限流，防止意外高额调用。

---

## 5. Python 工程

### 5.1 Protocol 与 ABC 各在什么时候使用？

Protocol（`typing.Protocol`）适合"鸭子类型"场景——只要对象有对应方法签名就满足接口，无需继承，适合测试 mock 和依赖注入。ABC（`abc.ABC` + `@abstractmethod`）适合需要共享实现或强制子类调用 super().__init__() 的场景。askbook 中对 Embedder、VectorStore 等外部依赖使用 Protocol 定义接口，对 Pipeline Node 等有层次结构的基类使用 ABC。

```python
class EmbedderProtocol(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
```

### 5.2 TypedDict 在 Pipeline Context 中的作用是什么？

TypedDict 为字典提供类型约束，让每个 key 有明确的类型注解。askbook 在 Pipeline Context 中使用 TypedDict 传递节点间的状态（query、retrieved chunks、rewritten query、answer 等），既保持了字典的灵活性（可扩展、可序列化），又获得了 IDE 类型检查和自动补全的好处。

### 5.3 Pydantic 的 validators 和 model_config 如何使用？

`model_config` 用于配置模型行为（如 `frozen=True` 强制不可变、`extra="forbid"` 禁止未知字段）。Validator（`@field_validator` 和 `@model_validator`）在校验阶段执行自定义逻辑，如字段归一化、跨字段依赖检查。askbook 的 Config 模型和 MCP I/O 模型广泛使用这些特性来保证数据一致性。

```python
class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    temperature: float = 0.1
    model_config = {"frozen": True, "extra": "forbid"}
```

### 5.4 异步 Trace 写入如何使用上下文管理器实现？

askbook 的 TraceWriter 使用 `@contextmanager` 与 `yield` 实现 span 生命周期管理。调用方通过 `with writer.span("node_name") as span:` 自动记录开始/结束时间和执行状态，无论代码是否抛出异常都能保证 span 正确关闭。内部使用 `contextvars` 维护 trace_id 的层级关系。

```python
@contextmanager
def span(self, node_name: str):
    span_id = str(uuid4())
    self._start_span(span_id, node_name)
    try:
        yield SpanContext(span_id)
    finally:
        self._end_span(span_id)
```

### 5.5 pytest 的 fixtures（tmp_path、monkeypatch）在测试中的典型用法是什么？

`tmp_path` 提供测试专用的临时目录，用于隔离 trace 文件、ChromaDB 数据等 IO 操作，避免测试间相互污染。`monkeypatch` 用于替换环境变量、配置值或外部依赖返回值，让测试不依赖真实服务。askbook 的测试将两者结合使用：`monkeypatch.setenv` 修改配置，`tmp_path` 作为 trace_dir 确保测试清理。

```python
def test_overview_page_renders(tmp_path: Path) -> None:
    cfg = Settings(observability=ObservabilityConfig(trace_dir=str(tmp_path / "traces")))
    at = AppTest.from_file(str(PAGES_DIR / "1_overview.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=20)
    assert not at.exception
```
