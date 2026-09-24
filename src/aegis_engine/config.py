"""Runtime configuration for the orchestration engine.

Configuration is intentionally small in the first milestone. Environment
variables make local development easy without introducing a configuration
framework or requiring secrets in source control.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when configuration cannot be used safely."""


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean value")


def _parse_positive_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


@dataclass(frozen=True)
class Settings:
    """Validated settings used by the model gateway."""

    ollama_base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen2.5:7b"
    request_timeout_seconds: float = 60.0
    max_output_tokens: int = 1024
    ollama_keep_alive: str = "10m"
    local_only: bool = True
    approved_file_roots: tuple[str, ...] = ()
    laya_enabled: bool = False
    laya_model: str | None = None
    laya_preload: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.ollama_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("ollama_base_url must be an absolute HTTP(S) URL")
        if not self.default_model.strip():
            raise ConfigurationError("default_model must not be empty")
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("request_timeout_seconds must be greater than zero")
        if self.max_output_tokens <= 0:
            raise ConfigurationError("max_output_tokens must be greater than zero")
        if not self.ollama_keep_alive.strip():
            raise ConfigurationError("ollama_keep_alive must not be empty")
        if self.laya_model is not None and not self.laya_model.strip():
            raise ConfigurationError("laya_model must not be empty when provided")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables without reading secrets."""

        values = os.environ if environ is None else environ
        timeout_value = values.get("AEGIS_REQUEST_TIMEOUT_SECONDS", "60")
        max_output_tokens_value = values.get("AEGIS_MAX_OUTPUT_TOKENS", "1024")
        local_only_value = values.get("AEGIS_LOCAL_ONLY", "true")
        laya_enabled_value = values.get("AEGIS_LAYA_ENABLED", "false")
        laya_preload_value = values.get("AEGIS_LAYA_PRELOAD", "false")
        roots = tuple(
            root.strip()
            for root in values.get("AEGIS_APPROVED_FILE_ROOTS", "").split(os.pathsep)
            if root.strip()
        )
        return cls(
            ollama_base_url=values.get("AEGIS_OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            default_model=values.get("AEGIS_DEFAULT_MODEL", cls.default_model),
            request_timeout_seconds=_parse_positive_float(
                timeout_value, name="AEGIS_REQUEST_TIMEOUT_SECONDS"
            ),
            max_output_tokens=_parse_positive_int(
                max_output_tokens_value, name="AEGIS_MAX_OUTPUT_TOKENS"
            ),
            ollama_keep_alive=values.get("AEGIS_OLLAMA_KEEP_ALIVE", cls.ollama_keep_alive),
            local_only=_parse_bool(local_only_value, name="AEGIS_LOCAL_ONLY"),
            approved_file_roots=roots,
            laya_enabled=_parse_bool(laya_enabled_value, name="AEGIS_LAYA_ENABLED"),
            laya_model=values.get("AEGIS_LAYA_MODEL") or None,
            laya_preload=_parse_bool(laya_preload_value, name="AEGIS_LAYA_PRELOAD"),
        )


settings = Settings.from_env()
