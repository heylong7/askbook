"""PII redaction utilities for observability traces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Compiled once at module level: (pattern, label)
# EMAIL must come before TOKEN so that email addresses are not partially matched
# as tokens.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        "EMAIL",
    ),
    (
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "PHONE",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{32,2048}(?![A-Za-z0-9+/])"),
        "TOKEN",
    ),
)


@dataclass(frozen=True)
class Redactor:
    """Apply PII redaction patterns to strings and nested mappings."""

    enabled: bool = True

    def apply(self, text: str) -> str:
        """Return *text* with all PII patterns replaced by redaction labels."""
        if not self.enabled:
            return text
        result = text
        for pattern, label in _PATTERNS:
            result = pattern.sub(f"<REDACTED:{label}>", result)
        return result

    def apply_to_mapping(self, m: Mapping[str, Any]) -> dict[str, Any]:
        """Recursively redact string values inside a nested mapping."""
        out: dict[str, Any] = {}
        for key, value in m.items():
            out[key] = self._redact_value(value)
        return out

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.apply(value)
        if isinstance(value, Mapping):
            return self.apply_to_mapping(value)
        if isinstance(value, (list, tuple)):
            return type(value)(self._redact_value(item) for item in value)
        return value
