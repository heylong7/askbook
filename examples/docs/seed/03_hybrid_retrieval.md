# Hybrid Retrieval and Query Pipeline

The Query Pipeline processes user questions through a chain of nodes: query rewriting, hybrid retrieval, reranking, and answer synthesis. The core innovation is the hybrid retrieval strategy combining sparse (BM25) and dense (BGE-M3) retrieval with Reciprocal Rank Fusion (RRF).

## Pipeline Nodes

```
Input: query_text, collection, top_k
  |
[QueryRewriterNode]       (default passthrough, optional LLM rewrite)
  |
[HyDENode]                (default disabled; generates hypothetical docs)
  |
[HybridRetrieverNode]     BM25 + BGE-M3 parallel retrieval -> top_k*2 candidates
  |
[RRFFusionNode]           RRF score fusion, k=60
  |
[CrossEncoderRerankNode]  bge-reranker-v2-m3 coarse rerank -> top_k
  |
[LLMFineRerankNode]       (default disabled; LLM-as-judge fine rerank)
  |
[AnswerSynthesizerNode]   Jinja2 prompt -> Answer with citations
```

## BM25 + Dense Retrieval

BM25 is a sparse lexical retrieval algorithm based on TF-IDF. It excels at exact keyword matching and handles rare terms well. BGE-M3 is a dense multilingual embedding model that captures semantic similarity, handling synonyms and paraphrases. Together they cover both lexical precision and semantic understanding.

## Reciprocal Rank Fusion (RRF)

RRF merges two ranked lists without requiring score normalization. The formula is:

```
score(d) = sum over all lists of 1 / (k + rank(d))
```

where k=60 is the standard smoothing constant. RRF only depends on ranking positions, making it robust for heterogeneous retrieval systems where absolute scores (BM25 vs cosine similarity) are not comparable. When one list ranks a document high and the other ranks it low, RRF produces a balanced compromise score.

## Two-Stage Reranking

The Cross-Encoder (bge-reranker-v2-m3) jointly encodes (query, passage) pairs for more accurate relevance scoring. Unlike Bi-Encoders which pre-compute passage embeddings, Cross-Encoders process each pair independently, making them slower but more accurate -- ideal for reranking a small candidate set. An optional second stage uses an LLM as a judge for fine-grained relevance assessment.
