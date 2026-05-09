# 系统架构

askbook 是模块化、可插拔的 RAG 系统，严格分离入库（写路径）和查询（读路径）。所有外部依赖（LLM、Embedder、VectorStore、Splitter、Reranker）均抽象为 Protocol，切换 provider 只需修改配置，不改代码。

## 核心设计原则

### 可插拔抽象层

系统通过 Protocol（结构化子类型）和 ABC（继承式）定义扩展点：

- **LLMProviderProtocol**：`complete()` 和 `acomplete()` 文本生成
- **EmbedderProtocol**：`embed_query()`、`embed_passage()`、`embed_batch()`
- **RerankerProtocol**：`rerank()` 相关性重排序
- **SplitterProtocol**：`split()` 文档分割
- **TraceWriterProtocol**：`span()` 上下文管理器，记录可观测数据
- **VectorStoreABC**：5 个抽象方法定义向量存储操作
- **BasePipelineNode**：模板方法模式，自动 trace 上报

新增 provider 只需实现对应 Protocol 并在 ServiceRegistry 中注册一行，现有代码无需修改。

### 读写分离

入库管线（写路径）和查询管线（读路径）完全独立。入库是同步 CLI 批处理，查询是在线推理管线（MCP 工具调用或 CLI 命令触发）。

### 目录结构

源码采用 `src/` 布局，按功能域组织：

```
src/askbook/
├── core/           接口、模型、异常、服务注册
├── config/         配置加载、schema、默认值
├── providers/      LLM 适配器（Ollama/Qwen、DashScope/Qwen、OpenAI、Anthropic）
├── embeddings/     Embedding 适配器（BGE-M3、OpenAI、DashScope）
├── rerankers/      Cross-Encoder 重排序器
├── splitters/      文档分割器（递归、语义）
├── vectorstores/   ChromaDB 适配器、BM25 索引
├── ingestion/      7 节点入库管线
├── query/          7 节点查询管线
├── mcp_server/     MCP stdio 服务端 + 工具处理器
├── observability/  异步 JSONL trace 写入器
├── evaluation/     数据集加载、运行器、指标
├── dashboard/      Streamlit 多页面看板
├── prompts/        Jinja2 提示词模板
└── utils/          工具函数
```

## 运行时数据布局

所有运行时文件位于 `~/.askbook/`：

```
~/.askbook/
├── chroma/         ChromaDB 持久化存储
├── bm25/           BM25 pickle 索引文件
├── traces/         JSONL trace 日志（按日期分片）
├── cache/          MarkItDown 和 HuggingFace 模型缓存
└── eval_runs/      评估结果快照
```
