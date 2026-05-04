# Harness Engineering

Harness engineering is a set of practices that protect AI agent systems from three systemic failure modes: Context Rot, Hallucinated Completion, and Model Drift. askbook embeds these protections into its data models, validators, and trace schema.

## Three Failure Modes

### Context Rot

When pipeline nodes pass excessive data between each other, the model's context window fills with irrelevant information, degrading instruction following. Protection: `RetrievalResult` only carries a snippet (max 200 chars) and chunk_id. Full document text must be lazily loaded via chunk_id when needed by the synthesizer.

### Hallucinated Completion

When retrieval returns empty results, the LLM may still fabricate a plausible-sounding answer. Protection: `ToolResponse` has a pydantic model_validator that rejects `status=success` with empty `source_ids`. The MCP tool must return `status=warning` when no results are found.

### Model Drift

Over multiple executions, query rewriting nodes may systematically shift query meaning without anyone noticing. Protection: Trace records both `original_query` and `rewritten_query`. The Dashboard provides a diff view comparing original vs rewritten queries across all traces.

## Ten Harness Practices

1. External persistent memory: JSONL traces, eval snapshots, git log
2. Deterministic verification: ruff, mypy, pytest golden CI gates
3. Atomic tasks with context refresh: each pipeline node is a pure function with clean TypedDict I/O
4. Sub-agent swarm: parallel LLM-judge evaluation with asyncio.gather + semaphore
5. Skill files: MCP system_prompt under 500 tokens, tools capped at 6
6. Guard rails and checkpoints: SHA256 validation at ingestion, score threshold at retrieval, source_ids assertion at synthesis
7. Handoffs: doc_id versioning, trace date sharding for cross-session continuity
8. Human-in-the-loop: seed_manual QA manual annotation, dashboard user feedback
9. Architecture constraints: unidirectional module dependency, Protocol enforcement
10. Garbage collection: orphan chunk scanner, interface signature consistency checker

## Anti-Pattern Checklist

- Never carry raw_text/full_content in RetrievalResult (use snippet + lazy load)
- Never return status=success with empty source_ids (use status=warning)
- Always record original_query alongside rewritten_query
- Never share mutable state between pipeline nodes (use TypedDict I/O)
- Keep MCP tools at 6 or fewer with single responsibilities
- Use parallel evaluation (asyncio.gather) for batch LLM-judge, never serial
