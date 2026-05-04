# Evaluation System

askbook's evaluation framework quantifies both retrieval quality and answer quality through a multi-level dataset strategy and four categories of metrics.

## Three-Level Dataset Strategy

### Level 1 -- seed_manual (20-30 items)

Hand-crafted QA pairs committed to the repository. Each item includes a question, ground truth answer, and relevant chunk IDs. Used for fast CI regression testing (under 2 minutes). All items are human-reviewed and tagged by category (retrieval, synthesis, bilingual, empty_source).

### Level 2 -- llm_generated

Auto-generated QA pairs using an LLM-based question generator. The generator extracts key passages from ingested documents and formulates questions. An LLM judge filters low-quality pairs. This level scales evaluation coverage without manual effort.

### Level 3 -- user_feedback

Real user queries and feedback collected through the Dashboard thumbs-up/down mechanism. Long-term accumulation of high-value evaluation samples from actual usage patterns.

## Four Metric Categories

### Retrieval Metrics (zero LLM cost)

- **Hit Rate**: fraction of queries where at least one relevant chunk appears in top-k results.
- **MRR (Mean Reciprocal Rank)**: average of 1/rank for the first relevant chunk.
- **Recall@K**: fraction of all relevant chunks retrieved in top-k results.
- **NDCG@K**: Normalized Discounted Cumulative Gain, rewarding relevant chunks at higher ranks.

All retrieval metrics are pure Python computations with no LLM API calls, making them fast and free to run in CI.

### Generation Metrics (Ragas)

- **faithfulness**: whether the answer can be supported by retrieved context (hallucination detection).
- **answer_relevancy**: whether the answer actually addresses the question.
- **context_precision**: whether retrieved chunks are relevant to the question.
- **context_recall**: whether all necessary information was retrieved.

### LLM Judge Metrics

Custom scoring using a local LLM (Qwen) for accuracy, relevance, and groundedness assessments.

### Cost and Latency Metrics

P50/P90/P99 latency percentiles, average token consumption per query, and estimated USD cost (using a per-model cost table).
