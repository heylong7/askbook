# Harness 工程

Harness 工程是一组保护 AI Agent 系统免受三类系统性故障影响的实践。askbook 将这些保护嵌入到数据模型、校验器和 Trace schema 中。

## 三类故障模式

### 上下文腐烂（Context Rot）

Pipeline 节点间传递过多数据时，模型上下文窗口被无关信息填满，指令遵循能力下降。

**防护**：`RetrievalResult` 仅携带 snippet（最多 200 字符）和 chunk_id。完整文档文本需通过 chunk_id 按需懒加载。

### 幻觉补全（Hallucinated Completion）

检索返回空结果时，LLM 仍可能编造看似合理的答案。

**防护**：`ToolResponse` 的 pydantic model_validator 拒绝 `status=success` 搭配空 `source_ids`。无结果时 MCP 工具必须返回 `status=warning`。

### 模型漂移（Model Drift）

多次执行中，查询改写节点可能系统性偏移查询语义而不被察觉。

**防护**：Trace 同时记录 `original_query` 和 `rewritten_query`。Dashboard 提供 Diff 视图对比历次 trace 中的原始查询与改写查询。

## 十项 Harness 实践

1. **外部持久化记忆**：JSONL trace、评估快照、git log
2. **确定性验证**：ruff + mypy + pytest 作为 CI golden gate
3. **原子任务 + 上下文刷新**：每个 Pipeline 节点是纯函数，TypedDict I/O
4. **子 Agent 集群**：并行 LLM Judge 评估（asyncio.gather + semaphore）
5. **技能文件**：MCP system_prompt 控制在 500 token 以内，工具不超过 6 个
6. **护栏与检查点**：入库 SHA256 校验、检索分数阈值、合成 source_ids 断言
7. **交接机制**：doc_id 版本化、trace 按日分片保证跨会话连续
8. **人机协同**：seed_manual 人工标注、Dashboard 用户反馈
9. **架构约束**：单向模块依赖、Protocol 强制解耦
10. **垃圾回收**：孤立 chunk 扫描、接口签名一致性检查

## 反模式清单

- 禁止在 RetrievalResult 中携带 raw_text/full_content（用 snippet + 懒加载）
- 禁止 status=success 搭配空 source_ids（用 status=warning）
- 必须同时记录 original_query 与 rewritten_query
- 禁止 Pipeline 节点间共享可变状态（用 TypedDict I/O）
- MCP 工具保持 6 个或更少，单一职责
- 批量 LLM Judge 评估用 asyncio.gather 并行，不用串行
