from dataclasses import dataclass

from aegis_engine.agents import AgentRequest, AgentRuntime, build_default_agents
from aegis_engine.config import Settings
from aegis_engine.models import (
    ChatRequest,
    ChatResponse,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
)


@dataclass
class FakeProvider:
    name: str = "fake"

    def list_models(self) -> list[ModelInfo]:
        return []

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(model=request.model, content="structured agent result")


def build_runtime() -> AgentRuntime:
    registry = ModelRegistry()
    registry.register(
        ModelInfo(
            model_id="agent-model",
            provider="fake",
            local=True,
            available=True,
            capabilities=frozenset({"completion"}),
        )
    )
    return AgentRuntime(
        ModelGateway([FakeProvider()]),
        registry,
        router=DeterministicModelRouter(registry),
        provider_settings=Settings(local_only=True),
    )


def test_default_agents_are_explicit_profiles() -> None:
    runtime = build_runtime()
    for agent in build_default_agents():
        runtime.register(agent)

    result = runtime.run("analysis", AgentRequest(task="Compare these options", context="A is local."))

    assert result.success is True
    assert result.output == "structured agent result"
    assert result.model == "fake/agent-model"


def test_agent_rejects_oversized_context() -> None:
    runtime = build_runtime()
    runtime.register(build_default_agents()[0].__class__(
        name="small",
        description="small test agent",
        system_prompt="Return a concise result.",
        max_input_chars=5,
    ))

    result = runtime.run("small", AgentRequest(task="too long"))

    assert result.success is False
    assert result.error == "agent input exceeds its limit"
