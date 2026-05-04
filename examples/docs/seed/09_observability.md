# Observability and Tracing

askbook implements full-chain observability through asynchronous JSONL tracing, without requiring external APM services like LangSmith or Grafana. Every pipeline node automatically reports span events with timing, token usage, and cost data.

## JSONL Trace Format

Trace events are appended to daily-sharded JSONL files at `~/.askbook/traces/YYYY-MM-DD.jsonl`. Each event is a self-contained JSON object:

- **trace_id**: UUID identifying a single pipeline execution
- **span_id**: unique identifier for this node execution
- **parent_span_id**: parent node (null for root)
- **event_type**: span_start, span_end, or error
- **node_name**: pipeline node identifier
- **timestamp_utc**: ISO 8601 timestamp
- **duration_ms**: elapsed time (span_end only)
- **tags**: node-specific metadata (token counts, chunk counts, model info)
- **prompt_template_hash**: SHA256 of the rendered prompt template, for tracking prompt changes
- **error**: exception message if the span failed

## Async Trace Writer

The AsyncTraceWriter uses a background daemon thread with an internal queue. Span events are enqueued during pipeline execution and written asynchronously to avoid blocking the hot path. An `atexit` handler ensures the queue is drained on process exit.

Key configuration options:
- `observability.enabled` (bool, default true): set to false to short-circuit all trace writing
- `observability.trace_dir` (path): JSONL file storage directory
- `observability.retention_days` (int, default 7): automatically delete trace files older than N days
- `observability.pii_redaction` (bool, default true): redact phone numbers, emails, and token strings before writing

## PII Redaction

Before writing to disk, trace events pass through a regex-based redaction filter that masks:
- Chinese mobile phone numbers (11 digits)
- Email addresses
- API token patterns

This prevents accidentally persisting sensitive user data in trace logs.

## Trace in Pipeline Nodes

BasePipelineNode's `__call__` method automatically creates a span context manager. Subclasses only implement `run()`; the base class handles span lifecycle (start, end, error). Optional `before_run()` and `after_run()` hooks allow nodes to attach custom tags to spans. The `original_query` field in QuerySpan preserves the user's input before any rewriting, enabling drift detection in the Dashboard's rewrite diff view.
