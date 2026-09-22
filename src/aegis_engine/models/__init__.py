"""Provider contracts and model adapters."""

from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelProvider,
    ProviderError,
    ToolCall,
    UsageMetadata,
)
from .ollama import OllamaProvider
from .gateway import ModelGateway, ModelGatewayError
from .registry import ModelRegistry, ModelRegistryError
from .router import DeterministicModelRouter, RouteDecision, RoutingError, RoutingRequest

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "ModelProvider",
    "OllamaProvider",
    "ModelGateway",
    "ModelGatewayError",
    "ModelRegistry",
    "ModelRegistryError",
    "ProviderError",
    "DeterministicModelRouter",
    "RouteDecision",
    "RoutingError",
    "RoutingRequest",
    "ToolCall",
    "UsageMetadata",
]
