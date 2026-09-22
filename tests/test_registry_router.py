import pytest

from mmo_engine.models import (
    DeterministicModelRouter,
    ModelInfo,
    ModelRegistry,
    RoutingError,
    RoutingRequest,
)


def discovered_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            model_id="qwen2.5:7b",
            provider="ollama",
            local=True,
            available=True,
            capabilities=frozenset({"completion", "tools"}),
        ),
        ModelInfo(
            model_id="deepseek-r1:8b",
            provider="ollama",
            local=True,
            available=True,
            capabilities=frozenset({"completion", "tools", "thinking"}),
        ),
        ModelInfo(
            model_id="nomic-embed-text:latest",
            provider="ollama",
            local=True,
            available=True,
            capabilities=frozenset({"embedding"}),
        ),
        ModelInfo(
            model_id="hosted-model",
            provider="hosted",
            local=False,
            available=True,
            capabilities=frozenset({"completion"}),
        ),
    ]


def test_registry_applies_explicit_configuration() -> None:
    registry = ModelRegistry()
    registry.register_many(discovered_models())

    configured = registry.configure(
        provider="ollama",
        model_id="qwen2.5:7b",
        capabilities={"completion", "tools", "general"},
        enabled=False,
    )

    assert configured.enabled is False
    assert configured.capabilities == frozenset({"completion", "tools", "general"})
    assert all(model.model_id != "qwen2.5:7b" for model in registry.available())


def test_router_prefers_configured_model_and_returns_fallbacks() -> None:
    registry = ModelRegistry()
    registry.register_many(discovered_models())
    router = DeterministicModelRouter(registry, preferred_model="qwen2.5:7b")

    decision = router.route(RoutingRequest(required_capabilities=frozenset({"tools"})))

    assert decision.selected.model_id == "qwen2.5:7b"
    assert decision.fallbacks[0].model_id == "deepseek-r1:8b"
    assert "preferred model" in decision.reason


def test_router_rejects_embedding_model_for_tool_request() -> None:
    registry = ModelRegistry()
    registry.register_many(discovered_models())
    router = DeterministicModelRouter(registry)

    with pytest.raises(RoutingError):
        router.route(RoutingRequest(required_capabilities=frozenset({"tools"}), model_id="nomic-embed-text:latest"))


def test_router_enforces_local_only_scope() -> None:
    registry = ModelRegistry()
    registry.register_many(discovered_models())
    router = DeterministicModelRouter(registry)

    with pytest.raises(RoutingError):
        router.route(
            RoutingRequest(
                required_capabilities=frozenset({"completion"}),
                local_only=True,
                model_id="hosted-model",
            )
        )


def test_router_reports_missing_capability() -> None:
    registry = ModelRegistry()
    registry.register_many(discovered_models())
    router = DeterministicModelRouter(registry)

    with pytest.raises(RoutingError, match="reasoning"):
        router.route(RoutingRequest(required_capabilities=frozenset({"reasoning"})))

