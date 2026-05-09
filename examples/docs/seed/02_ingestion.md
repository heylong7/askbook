# 文档入库管线

入库管线将原始文档（PDF、Markdown、DOCX、TXT）转换为可检索的 chunk，存入 ChromaDB 和 BM25 索引。这是一个同步 CLI 批处理流程，包含 7 个 Pipeline 节点。

## 管线节点

```
输入: source_path
  |
[DocumentLoaderNode]   → Document（MarkItDown 将各类格式转为 Markdown）
  |
[SplitterNode]         → list[Chunk]（递归字符分割 + 重叠窗口）
  |
[EnrichmentNode]       → list[Chunk]（多模态图片描述注入 + 元数据增强）
  |
[DedupNode]            → (new_chunks, stale_chunk_ids)
  |
[EmbeddingNode]        → list[Chunk]（稠密向量嵌入）
  |
[VectorStoreWriteNode] → 新 chunk 写入 Chroma，删除过期 chunk
  |
[BM25IndexUpdateNode]  → 更新持久化 BM25 pickle 索引
  |
输出: IngestionResult
```

## SHA256 去重

对每个 chunk 的内容计算 SHA256 哈希。已在目标 collection 中的 chunk 会被跳过（复用）。同一源文件在旧版本中存在但新版本中消失的 chunk 被标记为过期，从 ChromaDB 中软删除，防止僵尸数据积累。

关键 ID 生成规则：
- **doc_id**：SHA256(绝对路径 + mtime_ns)，保证重复入库幂等
- **chunk_id**：SHA256(doc_id + chunk_index + content)，全局唯一

## 分割策略

支持两种分割器（通过 `SplitterProtocol` 注册）：

- **RecursiveTextSplitter**：递归字符分割，默认 chunk_size=600、chunk_overlap=80。按段落 → 句子 → 词的优先级逐级切分
- **SemanticSplitter**：基于嵌入相似度的语义分割，当相邻句子的余弦相似度低于阈值时切分。支持中英文混合分句

## 多模态增强

EnrichmentNode 在入库时对文档中的图片进行多模态处理：
- 提取文档中的图片，调用视觉 LLM（需配置 `ingestion.enrich_llm`）生成文字描述
- 将描述注入到对应 chunk 的内容中，使图片信息可被文本检索命中
- 未配置 enrich_llm 时跳过图片处理，不阻塞入库

## 并发控制

三个独立信号量分别控制嵌入、视觉 LLM 和 Chroma 写入的并发数：
- `embed_concurrency`（默认 4）
- `vision_concurrency`（默认 2）
- `chroma_concurrency`（默认 4）
