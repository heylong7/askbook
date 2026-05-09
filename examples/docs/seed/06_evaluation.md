# 评估体系

askbook 的评估框架通过种子数据集和四类指标，量化检索质量和答案质量。

## 数据集策略

### 已有：seed_manual（20 条）

人工标注的 QA 对，随仓库提交。每条包含问题、参考答案和相关 chunk ID。用于 CI 快速回归测试（2 分钟内完成）。所有条目按类别标记（检索、合成、中英混合、空结果）。

### 规划中：llm_generated

LLM 自动生成 QA 对，从已入库文档中提取关键段落并生成问题，由 LLM 评委过滤低质量条目。实现后将大幅扩展评估覆盖面。

### 收集中：user_feedback

Dashboard 第 5 页的赞/踩按钮将用户反馈写入 `eval_runs/feedback.jsonl`。目前反馈仅做收集，尚未集成到评估流水线中。

## 四类指标

### 检索指标（零 LLM 成本）

纯 Python 计算，无 API 调用，适合 CI 快速验证：

- **Hit Rate**：top-k 结果中至少有一条相关 chunk 的查询占比
- **MRR（Mean Reciprocal Rank）**：首个相关 chunk 的排名倒数均值
- **Recall@K**：所有相关 chunk 在 top-k 中被检索到的比例
- **NDCG@K**：归一化折损累计增益，奖励排在高位的相关 chunk

### 生成指标（Ragas）

评估 LLM 合成的答案质量：

- **faithfulness**：答案是否可被检索上下文支撑（幻觉检测）
- **answer_relevancy**：答案是否实际回答了问题
- **context_precision**：检索到的 chunk 是否与问题相关
- **context_recall**：是否检索到了所有必要信息

### LLM Judge 指标

使用本地 LLM 对准确性、相关性和依据性进行自定义评分。并行执行（asyncio.gather + semaphore）。

### Harness 健康指标

四项工程层面的健康度量：

- **completion_rate**：任务完成率
- **retries_per_task**：每任务平均重试次数
- **pass_at_1**：一次通过率
- **cost_per_task**：单任务平均成本

## 运行评估

```bash
uv run askbook eval --collection my-docs
```
