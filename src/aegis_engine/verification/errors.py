"""Stable error categories used by recovery and observability code."""

from __future__ import annotations

from enum import StrEnum

from aegis_engine.models.base import ProviderError


class ErrorCategory(StrEnum):
    TRANSIENT = "transient"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    PERMISSION = "permission"
    MODEL_OUTPUT = "model_output"
    UNKNOWN = "unknown"


def classify_error(error: BaseException | str) -> ErrorCategory:
    """Classify without exposing exception details or trusting model text."""

    if isinstance(error, ProviderError):
        return ErrorCategory.TRANSIENT if error.retryable else ErrorCategory.PROVIDER_UNAVAILABLE
    text = str(error).lower()
    if "timeout" in text or "timed out" in text:
        return ErrorCategory.TIMEOUT
    if "permission" in text or "not permitted" in text or "confirmation" in text:
        return ErrorCategory.PERMISSION
    if "invalid" in text or "required" in text or "unsupported" in text:
        return ErrorCategory.INVALID_INPUT
    if "empty response" in text or "model output" in text:
        return ErrorCategory.MODEL_OUTPUT
    return ErrorCategory.UNKNOWN
