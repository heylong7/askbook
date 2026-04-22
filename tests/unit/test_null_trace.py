def test_null_trace_writer_is_runtime_checkable_protocol() -> None:
    from askbook.core.interfaces import TraceWriterProtocol
    from askbook.observability.null_trace import NullTraceWriter

    w = NullTraceWriter()
    assert isinstance(w, TraceWriterProtocol)


def test_null_trace_span_yields_span_with_name_and_attributes() -> None:
    from askbook.core.interfaces import TraceSpan
    from askbook.observability.null_trace import NullTraceWriter

    w = NullTraceWriter()
    with w.span("load") as span:
        assert isinstance(span, TraceSpan)
        assert span.name == "load"
        span.attributes["docs"] = 2
    w.flush()
