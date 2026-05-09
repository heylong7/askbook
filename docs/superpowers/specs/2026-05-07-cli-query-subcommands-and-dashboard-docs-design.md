# CLI query subcommands + Dashboard docs + Streamlit deprecation fix

**Date:** 2026-05-07
**Status:** approved

## Scope

- Add `query ask` and `query search` subcommands to CLI
- Fix Streamlit `use_container_width` deprecation warnings (2 sites)
- Correct README CLI examples and add detailed Dashboard documentation

## 1. CLI: `query` command group

### Current state

`cli.py` defines `query` as a single `@app.command()` that unconditionally runs the full RAG pipeline (retrieval + rerank + LLM synthesis + citations).

### Target state

`query` becomes a Typer group with two subcommands:

```
askbook query
├── ask     QUESTION   Full RAG pipeline (retrieve → rerank → LLM synthesize → cite)
└── search  QUESTION   Retrieval-only (dense + sparse + RRF + CE rerank, no LLM)
```

### Implementation

**`src/askbook/cli.py`:**
- Create `query_app = typer.Typer(name="query", help="...", no_args_is_help=True)`
- Define `query_app.command("ask")` → calls `run_query()`
- Define `query_app.command("search")` → calls `run_search()`
- Register with `app.add_typer(query_app, name="query")`

**`src/askbook/query/cli.py`:**
- Add `run_search()` function
- Reuses `HybridRetriever` + `CrossEncoderRerankNode` for retrieval
- Runs a minimal pipeline: retriever → fusion → CE rerank (no rewriter, no HyDE, no LLM rerank, no synthesizer)
- Output: chunk_id, score, snippet, source_path per result
- LLM is still constructed (needed for dependency injection) but not invoked

### No backward compatibility

The old `askbook query "question"` syntax is removed. Users must use `ask` or `search`.

## 2. Streamlit deprecation fix

Replace deprecated `use_container_width` parameter:

| File | Change |
|------|--------|
| `src/askbook/dashboard/pages/3_ingestion.py:77` | Drop `use_container_width=True` from `st.plotly_chart()` (defaults to stretch) |
| `src/askbook/dashboard/pages/4_trace_viewer.py:83` | Replace `use_container_width=True` with `width='stretch'` in `st.dataframe()` |

## 3. README updates

### 3a. Fix incorrect CLI commands

- Architecture diagram (line 29): update CLI line to show both subcommands
- Quick start step 6 (lines 86-89): replace incorrect `query ask`/`query search` with correct forms
- Align with actual CLI that now has these exact subcommands

### 3b. Expand Dashboard section

Add detailed subsection covering:

**Startup & config:**
- Port customization (`--port`)
- Config file passing (`--config`)
- Must run in separate terminal from MCP server

**Page-by-page component descriptions:**

| Page | Key components documented |
|------|--------------------------|
| 1. System Overview | 3 metric cards (query count, P50 latency, token total), health check list, data source (trace_dir last 1 day) |
| 2. Data Browser | Collection selector, document DataFrame, chunk content preview (first 200 chars) |
| 3. Pipeline Monitor | Two tabs: Ingestion timeline (DataFrame + Plotly Gantt), Query Rewrite audit (original vs rewritten side-by-side) |
| 4. Trace Viewer | Multi-filter (days/event type/node name), event DataFrame, single-row detail JSON |
| 5. Evaluation | Harness 4 health metrics (Completion Rate, Retries/Task, Pass@1, Cost/Task), retrieval metrics (Recall, MRR, Hit Rate, NDCG), user feedback collection |

### 3c. Add FAQ entry

Explain that Dashboard and MCP server each need their own terminal window — the JSON-RPC error occurs when `uv run askbook dashboard` is accidentally typed into the MCP server's stdin.

## Files changed

| File | Operation |
|------|-----------|
| `src/askbook/cli.py` | Modify — `query` command group |
| `src/askbook/query/cli.py` | Modify — add `run_search()` |
| `src/askbook/dashboard/pages/3_ingestion.py` | Modify — drop `use_container_width` |
| `src/askbook/dashboard/pages/4_trace_viewer.py` | Modify — `use_container_width` → `width='stretch'` |
| `README.md` | Modify — fix CLI commands, expand Dashboard section |

## Testing

- Unit test: add `test_query_search` and `test_query_ask` to `tests/unit/test_cli.py`
- Smoke test: run `askbook query search "test" --collection demo` and `askbook query ask "test" --collection demo`
- Verify Streamlit warning is gone on dashboard pages 3 and 4
