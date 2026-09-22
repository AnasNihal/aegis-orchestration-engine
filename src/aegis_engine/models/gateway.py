"""Provider catalog and model-aware inference gateway."""

from __future__ import annotations

from collections.abc import Iterable

from aegis_engine.models.base import ChatRequest, ChatResponse, ModelInfo, ModelProvider, ProviderError
from aegis_engine.models.registry import ModelRegistry, ModelRegistryError


class ModelGatewayError(ProviderError):
    """Raised when a provider cannot be selected for a model."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="gateway")


class ModelGateway:
    """Keep provider selection out of orchestration code."""

    def __init__(self, providers: Iterable[ModelProvider] = ()) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ModelProvider, *, replace_existing: bool = False) -> None:
        if provider.name in self._providers and not replace_existing:
            raise ModelGatewayError(f"provider is already registered: {provider.name}")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ModelGatewayError(f"provider is not registered: {name}") from exc

    def refresh(self, registry: ModelRegistry) -> None:
        """Discover models from every registered provider."""

        for provider in self._providers.values():
            registry.refresh(provider)

    def chat(self, model: ModelInfo, request: ChatRequest) -> ChatResponse:
        if request.model != model.model_id:
            raise ModelGatewayError("chat request model does not match routed model")
        return self.get(model.provider).chat(request)
