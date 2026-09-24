"""Controlled execution of specialized model agents.

Agents are profiles over the shared model gateway. They do not spawn agents,
execute arbitrary tools, or decide their own permissions. The orchestrator
owns delegation and receives a structured result from one bounded inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from aegis_engine.config import Settings, settings
from aegis_engine.models import ChatMessage, ChatRequest, ModelGateway, ModelRegistry, ProviderError
from aegis_engine.models.router import DeterministicModelRouter, RoutingRequest


class AgentRuntimeError(ValueError):
    """Raised when an agent request or definition is invalid."""


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"completion"}))
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    max_input_chars: int = 32_000

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip() or not self.system_prompt.strip():
            raise AgentRuntimeError("agent name, description, and system_prompt are required")
        if self.max_input_chars <= 0:
            raise AgentRuntimeError("agent max_input_chars must be positive")
        if "completion" not in self.capabilities:
            raise AgentRuntimeError("agent capabilities must include completion")


@dataclass(frozen=True)
class AgentRequest:
    task: str
    context: str = ""
    model_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise AgentRuntimeError("agent task must not be empty")


@dataclass(frozen=True)
class AgentResult:
    agent: str
    success: bool
    output: str = ""
    model: str | None = None
    error: str | None = None
    usage: object | None = None


class AgentRuntime:
    """Run registered agents through one shared, provider-neutral gateway."""

    def __init__(
        self,
        gateway: ModelGateway,
        registry: ModelRegistry,
        *,
        router: DeterministicModelRouter | None = None,
        provider_settings: Settings | None = None,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.router = router or DeterministicModelRouter(registry)
        self.settings = provider_settings or settings
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition, *, replace_existing: bool = False) -> None:
        if agent.name in self._agents and not replace_existing:
            raise AgentRuntimeError(f"agent is already registered: {agent.name}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentDefinition:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentRuntimeError(f"unknown agent: {name}") from exc

    def all(self) -> tuple[AgentDefinition, ...]:
        return tuple(self._agents[name] for name in sorted(self._agents))

    def run(self, name: str, request: AgentRequest) -> AgentResult:
        agent = self.get(name)
        combined = request.task if not request.context else f"{request.task}\n\nContext:\n{request.context}"
        if len(combined) > agent.max_input_chars:
            return AgentResult(agent=name, success=False, error="agent input exceeds its limit")
        try:
            decision = self.router.route(
                RoutingRequest(
                    required_capabilities=agent.capabilities,
                    local_only=self.settings.local_only,
                    model_id=request.model_id,
                )
            )
            response = self.gateway.chat(
                decision.selected,
                ChatRequest(
                    model=decision.selected.model_id,
                    messages=(
                        ChatMessage(role="system", content=agent.system_prompt),
                        ChatMessage(role="user", content=combined),
                    ),
                    max_tokens=self.settings.max_output_tokens,
                    keep_alive=self.settings.ollama_keep_alive,
                ),
            )
        except (ProviderError, ValueError, LookupError) as exc:
            return AgentResult(agent=name, success=False, error=str(exc))
        if response.tool_calls:
            return AgentResult(
                agent=name,
                success=False,
                model=f"{decision.selected.provider}/{decision.selected.model_id}",
                error="agent tool calls are not enabled in the bounded runtime",
            )
        if not response.content.strip():
            return AgentResult(
                agent=name,
                success=False,
                model=f"{decision.selected.provider}/{decision.selected.model_id}",
                error="agent returned an empty result",
            )
        return AgentResult(
            agent=name,
            success=True,
            output=response.content,
            model=f"{decision.selected.provider}/{decision.selected.model_id}",
            usage=response.usage,
        )


def build_default_agents() -> tuple[AgentDefinition, ...]:
    """Return safe agent profiles without creating independent model loops."""

    return (
        AgentDefinition(
            name="coding",
            description="Analyze code, identify defects, and propose implementation changes.",
            system_prompt=(
                "You are the Aegis coding specialist. Analyze only the supplied task and context. "
                "Return findings, risks, and a concise implementation plan. Do not claim that code "
                "was changed or tests passed unless the orchestrator provides that evidence."
            ),
        ),
        AgentDefinition(
            name="analysis",
            description="Compare evidence and produce a structured analysis.",
            system_prompt=(
                "You are the Aegis analysis specialist. Separate facts, assumptions, and conclusions. "
                "Use a clear structure and state when evidence is insufficient."
            ),
        ),
        AgentDefinition(
            name="research",
            description="Organize supplied research material and identify open questions.",
            system_prompt=(
                "You are the Aegis research specialist. Work only from supplied sources or context. "
                "Do not invent citations, browsing results, or facts that were not provided."
            ),
        ),
    )
