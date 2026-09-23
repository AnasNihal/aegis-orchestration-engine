"""Bounded single-step orchestration runtime.

This is deliberately the smallest useful engine: it routes one request to a
validated model, records state, and returns a structured task result. Optional
tool execution is bounded and injected behind this lifecycle rather than
embedded in the provider adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json

from aegis_engine.config import Settings, settings
from aegis_engine.decisions import (
    DecisionProvider,
    DecisionProviderError,
    LayaDecisionEngine,
    LayaTaskUnderstanding,
)
from aegis_engine.models.base import ChatMessage, ChatRequest, ProviderError
from aegis_engine.models.gateway import ModelGateway
from aegis_engine.models.router import DeterministicModelRouter, RoutingRequest
from aegis_engine.tasks.state import InMemoryTaskStateStore, TaskResult, TaskState, TaskStatus
from aegis_engine.tools.executor import ToolExecutor, ToolResult


@dataclass(frozen=True)
class OrchestratorConfig:
    max_retries: int = 1
    max_tool_iterations: int = 3

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be positive")


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
        tool_executor: ToolExecutor | None = None,
        decision_provider: DecisionProvider | None = None,
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.store = store or InMemoryTaskStateStore()
        self.settings = provider_settings or settings
        self.config = config or OrchestratorConfig()
        self.tool_executor = tool_executor
        self.decision_provider = decision_provider
        if self.decision_provider is None and self.settings.laya_enabled:
            self.decision_provider = LayaDecisionEngine(
                model=self.settings.laya_model,
                preload=self.settings.laya_preload,
            )

    def run(
        self,
        user_request: str,
        *,
        task_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        tool_names: tuple[str, ...] = (),
    ) -> TaskState:
        state = self.store.create(user_request, task_id=task_id)
        if is_cancelled and is_cancelled():
            return self._save(state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED))

        state = self._save(state.transition(TaskStatus.PLANNING, phase="routing"))
        if tool_names and self.tool_executor is None:
            return self._fail(state, "Tool execution is not configured for this task")
        understanding_completed = False
        if self.decision_provider is not None:
            state = self._save(state.with_updates(phase="task_understanding"))
            try:
                understanding = LayaTaskUnderstanding(self.decision_provider).analyze(user_request)
                state = self._save(
                    state.with_updates(
                        results=(
                            *state.results,
                            TaskResult(
                                step="understand_request",
                                success=True,
                                output=json.dumps(understanding.decisions, default=str),
                                model=f"{understanding.provider}/{understanding.model}",
                            ),
                        )
                    )
                )
                understanding_completed = True
            except DecisionProviderError as exc:
                # Understanding is advisory in this milestone. A missing or
                # failed optional Laya installation must not disable Ollama.
                state = self._save(
                    state.with_updates(errors=(*state.errors, f"task understanding: {exc}"))
                )
        try:
            tool_schemas = (
                self.tool_executor.registry.provider_schemas(set(tool_names))
                if self.tool_executor and tool_names
                else []
            )
        except Exception as exc:
            return self._fail(state, f"Tool selection failed: {exc}")
        try:
            required_capabilities = {"completion"}
            if tool_names:
                required_capabilities.add("tools")
            decision = self.router.route(
                RoutingRequest(
                    required_capabilities=frozenset(required_capabilities),
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

        messages = [ChatMessage(role="user", content=user_request)]
        for iteration in range(self.config.max_tool_iterations):
            if is_cancelled and is_cancelled():
                return self._save(
                    state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED)
                )
            response, state = self._chat_with_retries(
                state,
                model=model,
                messages=messages,
                tools=tool_schemas,
            )
            if response is None:
                return state
            if not response.tool_calls:
                if not response.content:
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
                        completed_steps=(
                            *(("understand_request",) if understanding_completed else ()),
                            "generate_response",
                        ),
                        results=(*state.results, result),
                        final_output=response.content,
                    ).transition(TaskStatus.COMPLETED, phase="complete")
                )
            if self.tool_executor is None:
                return self._fail(state, "Model requested tools but tool execution is not configured")
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            state = self._save(state.with_updates(phase=f"tool_iteration_{iteration + 1}"))
            for call in response.tool_calls:
                result = self.tool_executor.execute_call(call, allowed_tools=set(tool_names))
                state = self._record_tool_result(state, result, model=f"{model.provider}/{model.model_id}")
                if result.requires_confirmation:
                    return self._save(
                        state.with_updates(current_step=f"confirm:{call.name}").transition(
                            TaskStatus.WAITING_FOR_CONFIRMATION, phase="confirmation"
                        )
                    )
                messages.append(
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        content=json.dumps(
                            {"success": result.success, "output": result.output, "error": result.error},
                            default=str,
                        ),
                    )
                )
        return self._fail(state, "Tool iteration limit reached")

    def _chat_with_retries(self, state: TaskState, *, model, messages, tools):
        attempts = 0
        while True:
            try:
                response = self.gateway.chat(
                    model,
                    ChatRequest(model=model.model_id, messages=tuple(messages), tools=tuple(tools)),
                )
                return response, state
            except ProviderError as exc:
                attempts += 1
                will_retry = exc.retryable and attempts <= self.config.max_retries
                state = self._save(
                    state.with_updates(
                        retry_count=state.retry_count + (1 if will_retry else 0),
                        errors=(*state.errors, str(exc)),
                    )
                )
                if not will_retry:
                    return None, self._fail(state, str(exc))

    def _record_tool_result(self, state: TaskState, result: ToolResult, *, model: str) -> TaskState:
        output = json.dumps(result.output, default=str) if result.success else ""
        task_result = TaskResult(
            step=f"tool:{result.tool_name}",
            success=result.success,
            output=output,
            model=model,
            error=result.error,
        )
        errors = state.errors
        if result.error:
            errors = (*errors, f"{result.tool_name}: {result.error}")
        return self._save(
            state.with_updates(
                results=(*state.results, task_result),
                errors=errors,
            )
        )

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
