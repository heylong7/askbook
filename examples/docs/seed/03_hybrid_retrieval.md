# 混合检索与查询管线

查询管线将用户问题经过改写、混合检索、融合排序、重排序和答案合成等多个环节，最终返回带引用的答案。核心创新是 BM25 稀疏检索 + 稠密向量检索的混合策略，用 RRF 融合两路结果。

## 管线节点

```
输入: query_text, collection, top_k
  |
[QueryRewriterNode]        → （默认透传，可选 LLM 改写）
  |
[HyDENode]                 → （默认禁用；生成假设文档扩充查询）
  |
[HybridRetrieverNode]      → BM25 + 稠密向量并行检索，各取 top_k*2
  |
[RRFFusionNode]            → RRF 分数融合，k=60
  |
[CrossEncoderRerankNode]   → bge-reranker-v2-m3 粗排，截断到 top_k
  |
[LLMFineRerankNode]        → （默认禁用；LLM 精细重排序）
  |
[AnswerSynthesizerNode]    → Jinja2 模板 → LLM 合成 + 引用标注
```

## BM25 + 稠密检索

- **BM25**：基于 TF-IDF 的稀疏词法检索，擅长精确关键词匹配和处理低频术语
- **稠密向量**：BGE-M3 / text-embedding-v3 等多语言嵌入模型，捕捉语义相似度，处理同义词和改写
- 两路并行检索，各自返回 top_k*2 候选，合并后进入融合

## Reciprocal Rank Fusion（RRF）

RRF 合并两个排序列表，不依赖绝对分数：

```
score(d) = Σ 1 / (k + rank(d))
```

k=60 为标准平滑常数。RRF 仅依赖排序位置，适合 BM25 和余弦相似度这两种异构分数的融合。当两路对同一文档排名分歧时，RRF 给出折中结果。

## 两级重排序

- **Cross-Encoder（bge-reranker-v2-m3）**：联合编码 (query, passage) 对，精度高于双编码器。对融合后的候选集逐一打分，截断到 top_k
- **LLM Fine Rerank（可选）**：用 LLM 对 top_k 做精细相关性判断，默认关闭（需 `enable_llm_rerank: true`）

## HyDE（假设文档嵌入）

开启后（`enable_hyde: true`），LLM 先根据问题生成一篇假设回答文档，用该文档的嵌入向量替代原始问题进行检索。适合问题短、文档长的场景。
