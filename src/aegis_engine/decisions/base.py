"""Provider-neutral contracts for non-generative decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


DecisionState = str | Mapping[str, Any] | Sequence[Any] | None


class DecisionProviderError(RuntimeError):
    """A decision provider could not produce a safe result."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


@dataclass(frozen=True)
class DecisionRequest:
    """A state object and typed questions sent to a decision provider."""

    state: DecisionState
    questions: Mapping[str, Mapping[str, Any]]
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("questions must not be empty")


@dataclass(frozen=True)
class DecisionResponse:
    """Provider-normalized decision output."""

    provider: str
    model: str
    decisions: Mapping[str, Any]
    latency_ms: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecisionProvider(Protocol):
    """Minimal interface for typed decision engines."""

    name: str

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        """Answer the request's typed questions."""
