"""Integration tests for dashboard Page 5 — Evaluation + Harness metrics."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from askbook.observability.schema import TraceEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: str,
    node_name: str,
    status: str | None = None,
    retry: str | None = None,
    estimated_cost_cny: float = 0.0,
) -> TraceEvent:
    """Factory for a lightweight TraceEvent with the tags we care about."""
    tags: dict[str, object] = {}
    if status is not None:
        tags["status"] = status
    if retry is not None:
        tags["retry"] = retry
    if estimated_cost_cny:
        tags["estimated_cost_cny"] = estimated_cost_cny
    return TraceEvent(
        trace_id="t1",
        span_id="s1",
        event_type=event_type,
        node_name=node_name,
        timestamp_utc=datetime(2026, 5, 1, tzinfo=timezone.utc),  # noqa: UP017
        duration_ms=10.0,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDashboardPage5WithoutData:
    """Page 5 should render gracefully when no trace data or eval data exists."""

    def test_imports_and_runs_without_trace_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importing the page module should not crash when paths don't exist."""
        mock_settings = MagicMock()
        mock_settings.observability.trace_dir = "/nonexistent/traces"

        # Patch source modules *before* importing the page module so the
        # module-level code picks up the mocked functions.
        monkeypatch.setattr(
            "askbook.config.settings.load_settings",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "askbook.dashboard.loader.load_events",
            lambda trace_dir, days: [],
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr("pathlib.Path.glob", lambda self, pat: [])

        import importlib

        module_name = "askbook.dashboard.pages.5_evaluation"
        # Clear the cached module so monkeypatches take effect on import.
        sys.modules.pop(module_name, None)

        # Mock streamlit functions so the module-level code does not
        # require a running Streamlit runtime.
        monkeypatch.setattr("streamlit.title", MagicMock())
        monkeypatch.setattr("streamlit.header", MagicMock())
        monkeypatch.setattr(
            "streamlit.columns",
            lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))],
        )
        monkeypatch.setattr("streamlit.metric", MagicMock())
        monkeypatch.setattr("streamlit.expander", lambda title: MagicMock())
        monkeypatch.setattr("streamlit.markdown", MagicMock())
        monkeypatch.setattr("streamlit.info", MagicMock())

        # The module-level code runs on import — this should not crash.
        importlib.import_module(module_name)


class TestDashboardPage5WithEvents:
    """Page 5 should display harness metrics computed from trace events."""

    def test_shows_harness_metrics_from_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When load_events returns canned events, st.metric should be called with
        expected values — 8/12 success (66.7%), below the 95 % threshold."""
        mock_settings = MagicMock()
        mock_settings.observability.trace_dir = "/fake/traces"

        monkeypatch.setattr(
            "askbook.config.settings.load_settings",
            lambda: mock_settings,
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr("pathlib.Path.glob", lambda self, pat: [])

        # Canned events: 8 successful MCP calls out of 12, 2 retries
        canned = [
            _make_event("span_end", "ask", status="success", estimated_cost_cny=0.02),
            _make_event("span_end", "search", status="success"),
            _make_event("span_end", "ask", status="success", estimated_cost_cny=0.03),
            _make_event("span_end", "search", status="success"),
            _make_event("span_end", "ask", status="success", estimated_cost_cny=0.01),
            _make_event("span_end", "list_collections", status="success"),
            _make_event("span_end", "ask", status="success", estimated_cost_cny=0.04),
            _make_event("span_end", "ask", status="success", estimated_cost_cny=0.02),
            _make_event("span_end", "ask", status="error"),
            _make_event("span_end", "search", status="error"),
            _make_event("span_end", "ask", retry="true"),
            _make_event("span_end", "search", retry="true"),
        ]
        monkeypatch.setattr(
            "askbook.dashboard.loader.load_events",
            lambda trace_dir, days: canned,
        )

        import importlib

        module_name = "askbook.dashboard.pages.5_evaluation"
        sys.modules.pop(module_name, None)

        monkeypatch.setattr("streamlit.title", MagicMock())
        monkeypatch.setattr("streamlit.header", MagicMock())
        monkeypatch.setattr(
            "streamlit.columns",
            lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))],
        )
        mock_metric = MagicMock()
        monkeypatch.setattr("streamlit.metric", mock_metric)
        monkeypatch.setattr("streamlit.expander", lambda title: MagicMock())
        monkeypatch.setattr("streamlit.markdown", MagicMock())
        monkeypatch.setattr("streamlit.info", MagicMock())

        importlib.import_module(module_name)

        # 8/12 = 66.7%, below the 95% threshold → delta is negative, color inversed
        mock_metric.assert_any_call(
            "Completion Rate",
            "66.7%",
            delta="-28.3%",
            delta_color="inverse",
        )

    def test_threshold_delta_when_below_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When completion_rate is 40%, delta should reflect how far below 95%."""
        mock_settings = MagicMock()
        mock_settings.observability.trace_dir = "/fake/traces"

        monkeypatch.setattr(
            "askbook.config.settings.load_settings",
            lambda: mock_settings,
        )
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.setattr("pathlib.Path.glob", lambda self, pat: [])

        # Only 4 successes out of 10 — well below 95%
        canned = [_make_event("span_end", "ask", status="success") for _ in range(4)]
        for _ in range(6):
            canned.append(_make_event("span_end", "ask", status="error"))

        monkeypatch.setattr(
            "askbook.dashboard.loader.load_events",
            lambda trace_dir, days: canned,
        )

        import importlib

        module_name = "askbook.dashboard.pages.5_evaluation"
        sys.modules.pop(module_name, None)

        monkeypatch.setattr("streamlit.title", MagicMock())
        monkeypatch.setattr("streamlit.header", MagicMock())
        monkeypatch.setattr(
            "streamlit.columns",
            lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))],
        )
        mock_metric = MagicMock()
        monkeypatch.setattr("streamlit.metric", mock_metric)
        monkeypatch.setattr("streamlit.expander", lambda title: MagicMock())
        monkeypatch.setattr("streamlit.markdown", MagicMock())
        monkeypatch.setattr("streamlit.info", MagicMock())

        importlib.import_module(module_name)

        # 4/10 = 40%, well below 95% → delta is -55.0 %
        mock_metric.assert_any_call(
            "Completion Rate",
            "40.0%",
            delta="-55.0%",
            delta_color="inverse",
        )
