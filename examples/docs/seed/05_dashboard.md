# Streamlit Dashboard

askbook ships with a built-in Streamlit dashboard for system monitoring, data browsing, and trace inspection. The dashboard reads from ChromaDB metadata and JSONL trace files without requiring any external database or APM service.

## Running the Dashboard

```bash
uv run askbook dashboard           # default port 8501
uv run askbook dashboard --port 9000
```

Open `http://localhost:8501` in a browser to access the five-page dashboard.

## Page 1 -- System Overview

Displays real-time health badges for LLM, ChromaDB, and BM25 components. Shows today's query count, P50 latency, total token consumption, and estimated cost. A 24-hour trend chart plots query volume over time. Collection-level chunk counts are also displayed.

## Page 2 -- Data Browser

Read-only exploration of ChromaDB collections. Select a collection to view its document list (doc_id, source file path, chunk count, ingestion timestamp). Click a document to expand and preview individual chunk contents.

## Page 3 -- Ingestion Monitor

Lists historical ingestion runs parsed from JSONL traces. Each run shows document count, chunk statistics, and total duration. A Plotly Gantt chart visualizes per-node timing within each ingestion pipeline execution, making it easy to identify bottlenecks.

## Page 4 -- Trace Viewer

Parses the last 7 days of JSONL trace files to display query traces. Each trace shows latency, token usage, and estimated cost. Click to expand a waterfall view of all pipeline nodes and their individual durations, useful for diagnosing slow queries.

## Page 5 -- Evaluation Panel

Reads evaluation run snapshots from `.askbook/eval_runs/`. Displays historical metric trends as line charts (Recall@5, MRR, NDCG, faithfulness, answer relevancy). A button triggers new evaluation runs by calling the CLI as a subprocess.

## Caching Strategy

The dashboard uses `@st.cache_data` with JSONL file mtime as the cache key. When new trace data is written, the mtime changes and the cache auto-invalidates on the next refresh. This provides near-real-time updates without polling overhead.
