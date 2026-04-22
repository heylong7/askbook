"""Settings: YAML + .env + environment variables, merged in precedence order.

Precedence (highest wins):
    1. Environment variables (ASKBOOK_*)
    2. .env file
    3. User YAML (passed via --config)
    4. Packaged defaults.yaml
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from askbook.config.schema import (
    EmbeddingConfig,
    IngestionConfig,
    LLMConfig,
    MCPConfig,
    ObservabilityConfig,
    QueryConfig,
    VectorStoreConfig,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASKBOOK_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Path("~/.askbook")
    log_level: str = "INFO"

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("data_dir")
    @classmethod
    def _expand_data_dir(cls, v: Path) -> Path:
        return Path(str(v)).expanduser()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # env > dotenv > yaml/defaults (init_settings) > file secrets
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)


def _read_defaults() -> dict[str, Any]:
    defaults_text = (
        resources.files("askbook.config")
        .joinpath("defaults.yaml")
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(defaults_text) or {}


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings merging packaged defaults + user YAML + env vars."""
    data = _read_defaults()
    if config_path is not None:
        data = _deep_merge(data, _read_yaml(Path(config_path)))
    return Settings(**data)


__all__ = ["Settings", "load_settings"]
