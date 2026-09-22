"""Provider-neutral contracts for model inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable


class ProviderError(RuntimeError):
    """A provider could not complete an operation."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None

    def as_dict(self) -> dict[str, str]:
        message = {"role": self.role, "content": self.content}
        if self.name:
            message["name"] = self.name
        return message


@dataclass(frozen=True)
class ToolCall:
    """A provider-normalized request for a registered tool."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.2
    max_tokens: int | None = None
    tools: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass(frozen=True)
class UsageMetadata:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_ns: int | None = None


@dataclass(frozen=True)
class ChatResponse:
    model: str
    content: str
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    usage: UsageMetadata | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    """Model metadata discovered from or supplied to a provider.

    Capabilities are deliberately empty unless a provider or configuration
    explicitly establishes them. The engine must not infer tool support from
    a model name.
    """

    model_id: str
    provider: str
    local: bool
    available: bool
    enabled: bool = True
    capabilities: frozenset[str] = field(default_factory=frozenset)
    context_window: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    """Minimal interface the orchestrator will use for inference."""

    name: str

    def list_models(self) -> list[ModelInfo]:
        """Return models currently visible to this provider."""

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate one non-streaming response for a chat request."""
