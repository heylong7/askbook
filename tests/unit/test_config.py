"""Config system: YAML + .env + env override precedence, validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_defaults_yaml_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from askbook.config import load_settings

    for key in list(__import__("os").environ):
        if key.startswith("ASKBOOK_"):
            monkeypatch.delenv(key, raising=False)

    settings = load_settings()
    assert settings.log_level == "INFO"
    assert settings.llm.provider


def test_yaml_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from askbook.config import load_settings

    for key in list(__import__("os").environ):
        if key.startswith("ASKBOOK_"):
            monkeypatch.delenv(key, raising=False)

    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        "log_level: DEBUG\nllm:\n  provider: openai\n  model: gpt-4o-mini\n",
        encoding="utf-8",
    )
    settings = load_settings(config_path=yaml_path)
    assert settings.log_level == "DEBUG"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from askbook.config import load_settings

    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("log_level: INFO\n", encoding="utf-8")
    monkeypatch.setenv("ASKBOOK_LOG_LEVEL", "WARNING")
    settings = load_settings(config_path=yaml_path)
    assert settings.log_level == "WARNING"


def test_missing_required_field_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from askbook.config import load_settings

    for key in list(__import__("os").environ):
        if key.startswith("ASKBOOK_"):
            monkeypatch.delenv(key, raising=False)

    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  temperature: not-a-float\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_path=bad)


def test_data_dir_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    from askbook.config import load_settings

    for key in list(__import__("os").environ):
        if key.startswith("ASKBOOK_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("ASKBOOK_DATA_DIR", "~/.askbook-test")
    settings = load_settings()
    assert str(settings.data_dir).startswith(str(Path.home()))


def test_nested_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from askbook.config import load_settings

    monkeypatch.setenv("ASKBOOK_LLM__PROVIDER", "deepseek")
    settings = load_settings()
    assert settings.llm.provider == "deepseek"
