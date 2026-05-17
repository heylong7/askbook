# 面试题库 — askbook RAG 系统

## 目录

1. [RAG 基础](#1-rag-基础)
2. [MCP 协议](#2-mcp-协议)
3. [质量评估](#3-质量评估)
4. [Harness 工程](#4-harness-工程)
5. [Python 工程](#5-python-工程)
6. [系统设计与选型](#6-系统设计与选型)
7. [故障排查与可靠性](#7-故障排查与可靠性)

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

### 1.6 RRF 融合的公式和核心优势是什么？

RRF（Reciprocal Rank Fusion）公式：`score = Σ 1 / (k + rank_i)`，k 通常取 60。

核心优势：
- **量纲无关**：BM25 使用 TF-IDF 量纲，Dense 使用余弦相似度量纲，两者分数不可直接相加。RRF 只看排名位置不看分数值，天然解决融合问题
- **抗异常高分**：单路某文档分数虚高（如 BM25 遇到高频词），不会主导最终排序（k 常数平滑头部排名差异）
- 缺点：丢失分数的绝对大小信息，无法区分"第 1 名很相关"和"第 1 名勉强相关"

### 1.7 Cross-Encoder 与 Bi-Encoder 的本质区别是什么？

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| 编码方式 | query 和 doc 独立编码为向量 | query+doc 拼接后一起输入模型 |
| 交互深度 | 压缩为单一向量，丢失 token 级交互 | Attention 跨 query/doc 做 token 级交互 |
| 预计算 | 可离线预计算所有文档向量 | 每个 (query, doc) 对需实时前向推理 |
| 适用阶段 | 粗排（召回 top-100） | 精排（Rerank 取 top-5~10） |
| 计算量 | O(1) — 仅计算一次 query 向量 | O(N) — N 个候选 = N 次推理 |

工程实践：Bi-Encoder 粗召回 → Cross-Encoder 精排，askbook 默认使用 BGE-Reranker-v2-m3 做精排。

### 1.8 Hybrid Search 与 Cascade Retrieval 有什么区别？

- **Hybrid Search**：并行同时调用 Dense + BM25 双路检索，在库内用 RRF 融合排序，数据一致性强（原子操作），但每次都消耗完整 cost
- **Cascade Retrieval**：串行分层，先尝试 BM25（$0），效果不达标再触发 Dense（可设自适应阈值），成本优势明显——70% 请求只走 BM25

askbook 使用 Hybrid Search 方案，两路并行 + RRF 融合。

### 1.9 如何判断知识库中有无可靠答案？

多信号融合判断：
1. **Cross-Encoder Reranker 分数**：已有精排环节，直接看 top-1 reranker 分数，低于阈值（如 0.3）判定无可靠答案（q(doc, query) 交互比 Bi-Encoder 单一向量更有区分度）
2. **LLM 约束生成**：prompt 硬约束——"仅根据以下文档回答问题，若文档中无相关内容请直接回复'知识库中无相关内容'"
3. **组合策略**：reranker < 阈值直接拒绝；边界区域交由 LLM 判断

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

### 2.4 Claude Desktop / Claude Code 如何与 MCP Server 集成？

用户配置 MCP Server 名称、启动命令和环境变量。Client 以子进程方式启动 MCP Server，通过 stdio 交换 JSON-RPC 消息。

完整调用链路：
```
用户提问
  → Client 读取 MCP 配置（.mcp.json / claude_desktop_config.json）
  → 连接 MCP Server，发送 tools/list 请求获取工具清单
  → LLM 根据工具的 description 和 inputSchema 判断是否需要调用
  → 发送 tools/call 请求，传入参数
  → Server 执行工具（如混合检索），返回结构化结果
  → LLM 将结果融入回答生成
```

### 2.5 MCP Transport：stdio 与 HTTP 的区别是什么？

stdio transport 以子进程方式运行，通过标准输入输出通信，适合本地开发（无需网络配置），但每个工具调用共享同一进程生命周期。HTTP transport（SSE）基于 Server-Sent Events，支持远程调用和独立部署，但需要处理端口冲突和鉴权。askbook 默认使用 stdio，支持通过配置切换到 HTTP SSE。

### 2.6 MCP 工具的参数验证链路如何设计？

三层验证分工：

```
Client 侧（类型+范围检查）
  → Pydantic Schema 自动验证（字段类型/范围/必填，超出自动返回 422）
  → 业务逻辑验证（SQL 注入/XSS 防护、语义合法性）
```

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict = Field(default_factory=dict)

@mcp.tool()
def query_knowledge_hub(request: QueryRequest) -> str:
    # 业务逻辑安全验证
    sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UNION', 'SELECT']
    if any(kw.upper() in request.query.upper() for kw in sql_keywords):
        raise ValueError("SQL injection detected")
    return search(request.query, request.top_k, request.filters)
```

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

### 3.6 RAGAS 四大指标各自衡量什么？

| 指标 | 衡量什么 | 是否需要 Ground Truth |
|------|----------|----------------------|
| **Faithfulness** | 回答中每条陈述是否有检索文档支撑（防幻觉） | 否 |
| **Answer Relevance** | 回答是否真正回答了问题（从回答反向生成问题与原问题对比） | 否 |
| **Context Precision** | 检索返回的 chunk 是否相关，且相关的是否排在前面 | 否 |
| **Context Recall** | 检索到的内容是否覆盖了回答所需的全部信息 | **是** |

核心优势：前三个指标无需人工标注答案（无参考评估），大幅降低评估成本。

### 3.7 如何分析 RAGAS 指标矛盾？

常见矛盾场景：
- **高 Relevancy + 低 Faithfulness** → 生成侧幻觉。LLM 答得切题但编造了不存在的细节 → 降低温度、强化 prompt 约束、加后处理 NLI 验证
- **高 Faithfulness + 低 Relevancy** → 检索侧问题。chunk 内容可靠但不相关 → 检查 chunk 召回结果、加 Query Rewriting、换 embedding 模型

诊断流程：先看日志中实际召回的 chunk，判断是检索问题还是生成问题，分层排查。

### 3.8 MRR 的计算方式和使用场景是什么？

公式：`MRR = (1/N) * Σ (1 / rank_i)`

其中 rank_i 为第 i 个问题中第一个正确答案的排名位置。MRR = 0.82 意味着正确答案平均出现在第 1.2 位附近，检索质量较高。适用于每个问题只有一个正确答案的场景，不适合需要多个相关文档的场景（此时应用 NDCG）。

---

## 4. Harness 工程

### 4.1 什么是 Harness 工程？为什么它对 RAG 系统重要？

Harness 工程是 RAG 系统的质量保障层，涵盖监控、评估、成本控制和反模式检测。RAG 系统涉及多个组件（embedding、检索、重排序、LLM 生成），每个环节都可能引入质量退化——Harness 通过定义明确的 SLO（服务等级目标）和 CI 门禁来确保系统的可观测性和可维护性。

### 4.2 Completion Rate 监控什么？

Completion Rate 衡量 MCP 工具调用成功率（status=success 的调用数 / 总调用数），目标 >= 95%。它反映系统整体可用性——低于阈值可能表示 embedding 服务超时、ChromaDB 连接失败或 LLM 响应异常。askbook 从 JSONL Trace 中实时统计此指标并在 Dashboard 页面 5 用颜色编码呈现（达标绿色，超标红色）。

### 4.3 常见的 RAG 反模式有哪些？

askbook 的 Harness 层检测 6 种反模式：(1) Context Rot —— 检索上下文过时未更新；(2) Hallucinated Completion —— 答案包含上下文没有的信息；(3) Model Drift —— LLM 升级后行为不一致；(4) Empty Citation —— 成功响应但没有 source_id；(5) Zero-Recall Query —— 检索结果为空但 LLM 强行回答；(6) Token Overflow —— prompt 超过上下文窗口。每种反模式都在 CI 中做静态分析。

### 4.4 静态分析如何作为 CI 门禁？

静态分析在 CI 中检查 (1) Trace Schema 合规性（必填字段是否存在）；(2) ToolResponse 格式（source_ids 在 success 时必须非空）；(3) Baseline 回归（golden 指标下降超过阈值）。askbook 的 CI 门禁命令为 `ruff check && mypy --strict src/ && pytest --cov-fail-under=80`，确保代码质量符合 Harness 规范。

### 4.5 成本追踪与 Token 预算控制如何实现？

askbook 在每次 LLM 调用时估算 token 消耗并折算为人民币成本（基于模型 pricing），写入 Trace 的 `estimated_cost_cny` 标签。Dashboard 汇总每个任务的 Cost/Task 指标，阈值 <= ¥0.05。通过 `quota_per_hour` 和 `token_limit_per_call` 配置项做运行时限流，防止意外高额调用。

### 4.6 如何设计 RAG 系统的监控体系？

核心监控维度：
- **健康指标**（实时 Dashboard）：completion_rate >= 95%、retries_per_task <= 1.2、pass@1 >= 85%、cost_per_task <= ¥0.05
- **质量采样**：主动监控（随机采样 10-15% 请求跑 RAGAS）+ 被动反馈（用户标记问题 case 触发评估）
- **告警策略**：Hit Rate 下降超阈值触发告警，告警去重（同一问题 5 分钟内仅告警一次），分级（CRITICAL 立即通知 vs WARNING 记录）

### 4.7 实时告警系统的分级设计原则是什么？

生产级告警需包含：
- **告警去重/静默期**：同一问题 N 分钟内仅告警一次，避免轰炸
- **分级策略**：CRITICAL（立即 page + 自动诊断）vs WARNING（记录 + 聚合日报）vs INFO（减速通知）
- **自适应阈值**：基于正常值标准差动态调整，而非固定阈值（如 Hit Rate 通常波动 ±5%，固定 50% 阈值太宽松）
- **自动诊断**：告警触发时自动对比召回 chunk、检查各组件延迟、生成诊断报告

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

### 5.6 工厂模式如何在 askbook 中实现 Provider 切换？

askbook 通过 Registry + YAML 配置 + 动态创建实现 Provider 工厂：

- 抽象基类定义接口约定（Embedder、LLMProvider 等）
- Registry 字典维护 provider name → 具体实现的映射
- YAML 配置驱动选择（改 `embedding.provider: bge-m3` 即可切换），业务代码零修改

```python
class ProviderFactory:
    _registry = {
        "bge-m3": BGEEmbedding,
        "openai": OpenAIEmbedding,
        "dashscope": DashScopeEmbedding,
    }

    @staticmethod
    def create(config: dict):
        provider_type = config["provider"]
        if provider_type not in ProviderFactory._registry:
            raise ValueError(f"Unknown provider: {provider_type}")
        return ProviderFactory._registry[provider_type](**config.get("params", {}))
```

关键要素：抽象基类（接口约定）、Registry（动态查表）、错误处理（类型不存在时抛异常）、参数隔离（按 provider 分离配置）。

---

## 6. 系统设计与选型

### 6.1 ChromaDB 与其他向量数据库（Faiss、Qdrant、Milvus）的选型依据是什么？

| 向量库 | 定位 | 数据量 | 特点 |
|--------|------|--------|------|
| **Faiss** | 纯库，无持久化 | 研究场景 | 无服务端，需自行管理索引 |
| **ChromaDB** | 嵌入式，零运维 | < 10万文档 | 本地开发首选，开发效率高 |
| **Qdrant** | 中小规模生产 | 10万 ~ 1000万 | API 简洁、文档清晰、中小团队友好 |
| **Milvus** | 分布式大规模 | 1000万+ | 支持水平扩展，运维成本高 |

askbook 选择 ChromaDB 原因：本地部署场景为主、零运维需求、`pip install` 即可用。升级触发点：查询延迟 > 500ms 或数据量 > 50万文档。

### 6.2 Embedding 维度如何选择？1024 vs 3072 的权衡？

- 1024 维（BGE-M3）：适合本地部署，存储和查询开销小，10 万文档内可达满意 Hit Rate
- 3072 维（text-embedding-3-large）：精度更高，但存储空间增长 3 倍，查询速度（O(d)）约慢 3 倍
- OpenAI 官方数据：256 维丧失 1-2% Hit Rate，1024 vs 3072 丧失 <0.5%

决策公式：综合评估 Hit Rate + 延迟 + 存储成本。小规模本地 < 10 万文档 → 1024 维足够。当 Hit Rate < 0.5 且延迟可接受时考虑升维。

### 6.3 HNSW 向量索引的原理是什么？

HNSW（Hierarchical Navigable Small World）：
- **结构**：多层图结构，底层包含所有向量节点，每向上层随机采样越来越少节点
- **查询过程**：从最顶层稀疏图的入口点出发，找局部最近邻，逐层下降，每层缩小搜索范围，最终在底层精确检索
- **直觉类比**：类似跳表——顶层是"高速公路"快速定位区域，底层是"小路"精确抵达
- **复杂度**：O(log N)，远优于暴力搜索 O(N)
- **关键参数**：`ef_construction`（构建时搜索宽度）、`M`（每个节点最大连接数）
- **内存成本**：每个节点存储 M 条边 × 层数，1 亿节点时内存可能达 200GB+

ChromaDB 默认使用 HNSW 索引。

### 6.4 HNSW vs IVF 该如何选择？

| 维度 | HNSW | IVF |
|------|------|-----|
| 原理 | 分层图 + 贪心导航 | KMeans 聚类 + 倒排索引 |
| 查询复杂度 | O(log N) | O(N/K) — 仅搜索相关 cluster |
| 内存 | 大（每个节点存 M 条边） | 小（只存 K 个质心） |
| 适用 | < 1000 万，精度优先 | > 1000 万，内存可控 |
| 构建 | 慢（需构建多层图） | 快（KMeans 一次性聚类） |

结论：askbook 本地部署场景（< 10 万文档）用 HNSW（ChromaDB 默认）；扩展到千万级时考虑 Milvus 的 IVF 索引。

### 6.5 如何设计支持 100 并发的 RAG 服务架构？

三层架构设计：

**请求层**：FastAPI + async/await 异步处理非阻塞 I/O，Redis 队列削峰填谷，K8s Ingress 负载均衡

**LLM 推理层**（核心瓶颈）：部署 vLLM，核心技术是 Continuous Batching（请求完成立即移出 batch，新请求随时补入，GPU 不等最长序列）+ PagedAttention（KV Cache 分页管理，避免显存碎片）

**向量检索层**：ChromaDB → Milvus 分布式部署水平扩展，语义缓存（Semantic Cache）对高频相似 query 缓存结果

### 6.6 长对话上下文压缩的策略有哪些？

- **滑动窗口**：保留最近 N 轮对话（如 5 轮），超出截断
- **摘要压缩**：超出窗口的历史用 LLM 压缩为摘要，注入 system prompt（需保留决策点、核心问题、答案结论）
- **外部存储 + 按需加载**：元数据增强（日期、标题、摘要）+ 具体内容分离存储，LLM 判断需要时提取
- **Token 预算分配**：动态优先级 — query > generation > retrieved chunks > history

---

## 7. 故障排查与可靠性

### 7.1 当用户反馈"文档里有答案但系统回答不出来"时，如何系统排查？

RAG 全链路排查方法（分阶段验证）：

**1. 文档入库阶段** — 验证文档是否成功入库。检查 collection.get() 和 ingestion 日志。（常见问题：PDF 解析失败、编码问题）

**2. 分块阶段** — 检查答案所在段落是否被完整保留在某个 chunk 中。（常见问题：chunk_size 过小导致答案跨 chunk 截断、关键句在边界处被分割）

**3. Embedding 阶段** — 用答案所在 chunk 文本直接做 embedding，与 query embedding 计算余弦相似度。（常见问题：中英文混用、学术缩写 vs 全称）

**4. 召回阶段** — 打印 top-k 召回结果，确认目标 chunk 是否在其中。调大 top_k 看是否出现。（常见问题：top_k 过小、双路均未命中、RRF 权重不当）

**5. Rerank 阶段** — 确认 chunk 召回到了但 Rerank 后排名掉出截断位置。（常见问题：Reranker 对领域数据表现不佳、截断数量过小）

**6. LLM 生成阶段** — 将正确 chunk 直接塞入 prompt，LLM 能否正确回答。（常见问题：Context 过长导致"迷失在中间"、System Prompt 约束过严）

排查工具：askbook trace 日志、Dashboard 页面 4 Trace 查看器、RAGAS 分项指标定位瓶颈。

### 7.2 MCP 工具的可靠性设计包含哪些要素？

完整的 MCP 工具可靠性机制：
- **重试策略**：Exponential Backoff（1s → 2s → 4s），非等额间隔
- **熔断器模式**（Circuit Breaker）：连续失败 N 次后自动断路，停止发请求防止雪崩
- **超时分层**：向量库 500ms / LLM 30s / 整体请求 60s
- **可重试 vs 不可重试**：区分 4xx（参数问题，不重试）vs 5xx（服务端问题，重试）
- **降级链**：向量库不可用 → 降级到纯 BM25 → 返回"服务暂时不可用"

### 7.3 如何用隔离变量法排查 RAG 系统性能退化？

1. 用相同 query 对比本地 LLM vs 远程 API，隔离网络影响
2. 用不同长度 prompt 测试，确认 token 数量是否影响延迟
3. 对比相同 query 在不同时间点的 trace 数据（JSONL），找变化的环节
4. 对照实验：关闭 Query Rewriting / HyDE / Rerank 等节点逐一排除

关键思路：每次只改变一个变量，观察指标变化，缩小问题范围。

### 7.4 RAG 系统可靠性设计的全链路容错架构是什么？

| 层级 | 策略 | 说明 |
|------|------|------|
| **缓存** | Semantic Cache | 高频相似 query 0 cost 返回（一级防线） |
| **检索降级** | 向量库 → BM25 → 空结果 | 向量库不可用时自动降级 |
| **生成降级** | LLM A → LLM B → 仅返回文档 | Fallback 链 |
| **断路器** | 连续失败自动断路 | 防止雪崩 |
| **监控告警** | 4 项健康指标 + 6 种反模式 CI | 实时发现异常 |

