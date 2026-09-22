"""Model metadata registry used by deterministic routing."""

from __future__ import annotations

from dataclasses import replace

from mmo_engine.models.base import ModelInfo, ModelProvider


class ModelRegistryError(ValueError):
    """Raised when model metadata is invalid or ambiguous."""


class ModelRegistry:
    """In-memory registry for discovered and explicitly configured models."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelInfo] = {}

    def register(self, model: ModelInfo, *, replace_existing: bool = False) -> None:
        if not model.model_id.strip() or not model.provider.strip():
            raise ModelRegistryError("model_id and provider are required")
        key = (model.provider, model.model_id)
        if key in self._models and not replace_existing:
            raise ModelRegistryError(f"model is already registered: {model.provider}/{model.model_id}")
        self._models[key] = model

    def register_many(self, models: list[ModelInfo], *, replace_existing: bool = False) -> None:
        for model in models:
            self.register(model, replace_existing=replace_existing)

    def refresh(self, provider: ModelProvider) -> list[ModelInfo]:
        """Refresh availability while preserving explicit local metadata."""

        discovered = provider.list_models()
        refreshed: list[ModelInfo] = []
        for model in discovered:
            key = (model.provider, model.model_id)
            previous = self._models.get(key)
            if previous is not None:
                model = replace(
                    model,
                    enabled=previous.enabled,
                    capabilities=previous.capabilities or model.capabilities,
                    context_window=previous.context_window or model.context_window,
                )
            self._models[key] = model
            refreshed.append(model)
        return refreshed

    def configure(
        self,
        *,
        provider: str,
        model_id: str,
        capabilities: set[str] | frozenset[str] | None = None,
        enabled: bool | None = None,
        context_window: int | None = None,
    ) -> ModelInfo:
        """Apply explicit configuration to a registered model."""

        current = self.get(provider=provider, model_id=model_id)
        if capabilities is not None and any(not capability.strip() for capability in capabilities):
            raise ModelRegistryError("capabilities must contain non-empty names")
        updated = replace(
            current,
            capabilities=current.capabilities if capabilities is None else frozenset(capabilities),
            enabled=current.enabled if enabled is None else enabled,
            context_window=current.context_window if context_window is None else context_window,
        )
        self._models[(provider, model_id)] = updated
        return updated

    def get(self, *, provider: str, model_id: str) -> ModelInfo:
        try:
            return self._models[(provider, model_id)]
        except KeyError as exc:
            raise ModelRegistryError(f"unknown model: {provider}/{model_id}") from exc

    def all(self) -> tuple[ModelInfo, ...]:
        return tuple(sorted(self._models.values(), key=lambda item: (item.provider, item.model_id)))

    def available(self) -> tuple[ModelInfo, ...]:
        return tuple(model for model in self.all() if model.available and model.enabled)

