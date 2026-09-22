"""Bounded single-step orchestration runtime.

This is deliberately the smallest useful engine: it routes one request to a
validated model, records state, and returns a structured task result. Tool
execution and multi-step plans will be added behind this lifecycle rather than
embedded in the provider adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mmo_engine.config import Settings, settings
from mmo_engine.models.base import ChatMessage, ChatRequest, ProviderError
from mmo_engine.models.gateway import ModelGateway
from mmo_engine.models.registry import ModelRegistry
from mmo_engine.models.router import DeterministicModelRouter, RoutingRequest
from mmo_engine.tasks.state import InMemoryTaskStateStore, TaskResult, TaskState, TaskStatus


@dataclass(frozen=True)
class OrchestratorConfig:
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")


class Orchestrator:
    """Coordinate routing, bounded inference, and task lifecycle updates."""

    def __init__(
        self,
        gateway: ModelGateway,
        router: DeterministicModelRouter,
        *,
        store: InMemoryTaskStateStore | None = None,
        provider_settings: Settings | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.store = store or InMemoryTaskStateStore()
        self.settings = provider_settings or settings
        self.config = config or OrchestratorConfig()

    def run(
        self,
        user_request: str,
        *,
        task_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> TaskState:
        state = self.store.create(user_request, task_id=task_id)
        if is_cancelled and is_cancelled():
            return self._save(state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED))

        state = self._save(state.transition(TaskStatus.PLANNING, phase="routing"))
        try:
            decision = self.router.route(
                RoutingRequest(
                    required_capabilities=frozenset({"completion"}),
                    local_only=self.settings.local_only,
                )
            )
        except Exception as exc:
            return self._fail(state, f"Model routing failed: {exc}")

        model = decision.selected
        state = self._save(
            state.with_updates(
                plan=("generate_response",),
                current_step="generate_response",
                selected_models=(f"{model.provider}/{model.model_id}",),
            ).transition(TaskStatus.EXECUTING, phase="inference")
        )

        attempts = 0
        while True:
            if is_cancelled and is_cancelled():
                return self._save(
                    state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED)
                )
            try:
                response = self.gateway.chat(
                    model,
                    ChatRequest(
                        model=model.model_id,
                        messages=(ChatMessage(role="user", content=user_request),),
                    ),
                )
                if not response.content and not response.tool_calls:
                    return self._fail(state, "Model returned an empty response")
                result = TaskResult(
                    step="generate_response",
                    success=True,
                    output=response.content,
                    model=f"{model.provider}/{model.model_id}",
                )
                return self._save(
                    state.with_updates(
                        current_step=None,
                        completed_steps=("generate_response",),
                        results=(result,),
                        final_output=response.content,
                    ).transition(TaskStatus.COMPLETED, phase="complete")
                )
            except ProviderError as exc:
                attempts += 1
                state = self._save(
                    state.with_updates(
                        retry_count=min(attempts, self.config.max_retries),
                        errors=(*state.errors, str(exc)),
                    )
                )
                if not exc.retryable or attempts > self.config.max_retries:
                    return self._fail(state, str(exc))

    def _fail(self, state: TaskState, error: str) -> TaskState:
        result = TaskResult(step=state.current_step or "routing", success=False, error=error)
        errors = state.errors if state.errors and state.errors[-1] == error else (*state.errors, error)
        return self._save(
            state.with_updates(results=(*state.results, result), errors=errors).transition(
                TaskStatus.FAILED, phase="failed"
            )
        )

    def _save(self, state: TaskState) -> TaskState:
        return self.store.save(state)
