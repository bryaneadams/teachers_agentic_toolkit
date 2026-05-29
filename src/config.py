from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from errors import InputValidationError


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for one LLM provider profile."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    api_key_env: str | None = "OPENAI_API_KEY"
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestrationConfig:
    """Configuration for CrewAI orchestration behavior."""

    verbose: bool = False
    max_context_tokens: int = 6000


@dataclass(frozen=True)
class WorkflowConfig:
    """Runtime configuration for workflow execution."""

    llms: dict[str, LLMConfig] = field(default_factory=lambda: {"default": LLMConfig()})
    default_llm: str = "default"
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    env_file: Path = Path(".env")
    env_values: dict[str, str] = field(default_factory=dict, repr=False)

    def llm(self, name: str | None = None) -> LLMConfig:
        """Look up an LLM profile by name.

        Args:
            name (str | None, optional): Profile name to load, or None for the
                configured default profile. Defaults to None.

        Raises:
            InputValidationError: If the requested profile does not exist.

        Returns:
            LLMConfig: Matching LLM provider configuration.
        """
        key = name or self.default_llm
        try:
            return self.llms[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.llms))
            raise InputValidationError(
                f"Unknown LLM profile '{key}'. Available profiles: {available}"
            ) from exc

    def api_key_for(self, name: str | None = None) -> str | None:
        """Resolve the API key for an LLM profile.

        Args:
            name (str | None, optional): Profile name to resolve, or None for
                the configured default profile. Defaults to None.

        Returns:
            str | None: API key from loaded `.env` values or process
                environment, if configured.
        """
        return llm_api_key(self.llm(name), self.env_values)


def load_config(path: str | Path | None = None) -> WorkflowConfig:
    """Load workflow configuration from JSON, YAML, and `.env`.

    Args:
        path (str | Path | None, optional): Config file path. When omitted,
            default settings are used and `.env` is read from the current
            directory. Defaults to None.

    Raises:
        InputValidationError: If the config file path does not exist.

    Returns:
        WorkflowConfig: Parsed runtime configuration.
    """
    if path is None:
        env_file = Path(".env")
        return WorkflowConfig(env_file=env_file, env_values=load_dotenv(env_file))
    config_path = Path(path)
    if not config_path.exists():
        raise InputValidationError(f"Config file does not exist: {config_path}")

    raw = _load_mapping(config_path)
    llms = {
        name: LLMConfig(**_known_fields(values, LLMConfig))
        for name, values in raw.get("llms", {}).items()
    } or {"default": LLMConfig()}
    orchestration = OrchestrationConfig(
        **_known_fields(raw.get("orchestration", {}), OrchestrationConfig)
    )
    env_file = _resolve_env_file(raw.get("env_file", ".env"), config_path.parent)

    return WorkflowConfig(
        llms=llms,
        default_llm=raw.get("default_llm", "default"),
        orchestration=orchestration,
        env_file=env_file,
        env_values=load_dotenv(env_file),
    )


# Backward-compatible alias for callers that imported the old config name.
SlideMakerConfig = WorkflowConfig


def _known_fields(raw: dict[str, Any], config_type: type) -> dict[str, Any]:
    """Return only keys accepted by a dataclass config type."""

    allowed = {field.name for field in fields(config_type)}
    return {key: value for key, value in raw.items() if key in allowed}


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML config file as a mapping.

    Args:
        path (Path): Config file path.

    Raises:
        InputValidationError: If PyYAML is required but unavailable.
        InputValidationError: If the parsed config is not a mapping.

    Returns:
        dict[str, Any]: Parsed config data.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise InputValidationError(
                "Install PyYAML or use JSON config files."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise InputValidationError("Config must be a JSON/YAML object.")
    return data


def load_dotenv(path: str | Path) -> dict[str, str]:
    """Parse a simple `.env` file into key-value pairs.

    Args:
        path (str | Path): Path to the `.env` file.

    Raises:
        InputValidationError: If a non-empty line is not `KEY=VALUE`.
        InputValidationError: If an environment variable key is invalid.

    Returns:
        dict[str, str]: Parsed environment values, or an empty dict when the
            file does not exist.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise InputValidationError(
                f"Invalid .env line {line_number}: expected KEY=VALUE."
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise InputValidationError(f"Invalid .env key on line {line_number}: {key}")
        values[key] = _clean_env_value(value.strip())
    return values


def llm_api_key(
    config: LLMConfig, env_values: dict[str, str] | None = None
) -> str | None:
    """Resolve an API key for an LLM profile.

    Args:
        config (LLMConfig): LLM profile containing the environment variable
            name to read.
        env_values (dict[str, str] | None, optional): Parsed `.env` values to
            check before process environment. Defaults to None.

    Returns:
        str | None: API key value, or None when no key is configured or found.
    """
    if not config.api_key_env:
        return None
    env_values = env_values or {}
    return env_values.get(config.api_key_env) or os.getenv(config.api_key_env)


def _resolve_env_file(value: str | Path, config_dir: Path) -> Path:
    """Resolve a configured `.env` path relative to a config file.

    Args:
        value (str | Path): Path value from config.
        config_dir (Path): Directory containing the config file.

    Returns:
        Path: Absolute path, config-relative path when present, or the raw path.
    """
    env_path = Path(value)
    if env_path.is_absolute():
        return env_path
    candidate = config_dir / env_path
    if candidate.exists():
        return candidate
    return env_path


def _clean_env_value(value: str) -> str:
    """Normalize a `.env` value by removing quotes and inline comments.

    Args:
        value (str): Raw value text after the `=` sign.

    Returns:
        str: Cleaned value.
    """
    if not value:
        return ""
    if value[0] == value[-1:] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.split(" #", 1)[0].strip()
