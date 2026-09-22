"""Deterministic model routing rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from mmo_engine.models.base import ModelInfo
from mmo_engine.models.registry import ModelRegistry


class RoutingError(LookupError):
    """Raised when no model satisfies a routing request."""


@dataclass(frozen=True)
class RoutingRequest:
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    local_only: bool = True
    provider: str | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    selected: ModelInfo
    reason: str
    fallbacks: tuple[ModelInfo, ...] = ()


class DeterministicModelRouter:
    """Select models using explicit metadata and stable ordering only."""

    def __init__(self, registry: ModelRegistry, *, preferred_model: str | None = None) -> None:
        self.registry = registry
        self.preferred_model = preferred_model

    def route(self, request: RoutingRequest) -> RouteDecision:
        candidates = list(self.registry.available())
        if request.provider:
            candidates = [model for model in candidates if model.provider == request.provider]
        if request.local_only:
            candidates = [model for model in candidates if model.local]
        if request.required_capabilities:
            candidates = [
                model
                for model in candidates
                if request.required_capabilities <= model.capabilities
            ]
        if request.model_id:
            candidates = [model for model in candidates if model.model_id == request.model_id]

        if not candidates:
            requirements = ", ".join(sorted(request.required_capabilities)) or "no special capability"
            scope = "local " if request.local_only else ""
            raise RoutingError(f"No available {scope}model satisfies: {requirements}")

        candidates.sort(key=lambda model: (model.provider, model.model_id))
        if self.preferred_model:
            candidates.sort(key=lambda model: model.model_id != self.preferred_model)

        selected = candidates[0]
        if request.model_id:
            reason = "explicit model selection"
        elif self.preferred_model == selected.model_id:
            reason = "configured preferred model satisfies the request"
        elif request.required_capabilities:
            reason = "first available model with the required explicit capabilities"
        else:
            reason = "first available model in stable provider/model order"
        return RouteDecision(selected=selected, reason=reason, fallbacks=tuple(candidates[1:]))

