from dataclasses import dataclass

import pytest

from aegis_engine.config import Settings
from aegis_engine.decisions import DecisionRequest, DecisionResponse
from aegis_engine.models import (
    ChatRequest,
    ChatResponse,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
    ProviderError,
    ToolCall,
)
from aegis_engine.orchestration import Orchestrator, OrchestratorConfig
from aegis_engine.tasks import InMemoryTaskStateStore, TaskStateError, TaskStatus
from aegis_engine.tasks import PendingToolAction
from aegis_engine.tools import (
    PermissionClass,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    build_builtin_registry,
)


@dataclass
class FakeProvider:
    name: str = "fake"
    calls: int = 0
    failures_before_success: int = 0
    issue_tool_call: bool = False

    def list_models(self) -> list[ModelInfo]:
        return []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ProviderError("temporary provider failure", provider=self.name, retryable=True)
        if self.issue_tool_call and self.calls == 1:
            return ChatResponse(
                model=request.model,
                content="",
                tool_calls=(ToolCall(name="calculator", arguments={"expression": "2 + 2"}),),
            )
        return ChatResponse(model=request.model, content="completed locally")


@dataclass
class FakeDecisionProvider:
    name: str = "fake-decision"
    calls: int = 0

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        self.calls += 1
        return DecisionResponse(
            provider=self.name,
            model=request.model or "fake-decision-model",
            decisions={"domain": {"label": "general"}},
        )


def build_orchestrator(
    provider: FakeProvider,
    *,
    max_retries: int = 1,
    tool_executor: ToolExecutor | None = None,
    max_tool_iterations: int = 3,
    decision_provider: FakeDecisionProvider | None = None,
):
    registry = ModelRegistry()
    registry.register(
        ModelInfo(
            model_id="local-test-model",
            provider="fake",
            local=True,
            available=True,
            capabilities=frozenset({"completion", "tools"}),
        )
    )
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry)
    return Orchestrator(
        gateway,
        router,
        provider_settings=Settings(local_only=True),
        config=OrchestratorConfig(max_retries=max_retries, max_tool_iterations=max_tool_iterations),
        tool_executor=tool_executor,
        decision_provider=decision_provider,
    )


def test_orchestrator_completes_and_records_selected_model() -> None:
    provider = FakeProvider()
    orchestrator = build_orchestrator(provider)

    state = orchestrator.run("Summarize this task")

    assert state.status is TaskStatus.COMPLETED
    assert state.final_output == "completed locally"
    assert state.selected_models == ("fake/local-test-model",)
    assert state.completed_steps == ("generate_response",)
    assert [(step.name, step.status.value) for step in state.steps] == [
        ("generate_response", "completed")
    ]
    assert provider.calls == 1


def test_orchestrator_retries_only_bounded_retryable_failures() -> None:
    provider = FakeProvider(failures_before_success=1)
    orchestrator = build_orchestrator(provider, max_retries=1)

    state = orchestrator.run("Retry this task")

    assert state.status is TaskStatus.COMPLETED
    assert state.retry_count == 1
    assert provider.calls == 2


def test_orchestrator_records_optional_task_understanding_before_generation() -> None:
    provider = FakeProvider()
    decision_provider = FakeDecisionProvider()
    orchestrator = build_orchestrator(provider, decision_provider=decision_provider)

    state = orchestrator.run("Explain this Python code")

    assert state.status is TaskStatus.COMPLETED
    assert state.completed_steps == ("understand_request", "generate_response")
    assert [result.step for result in state.results] == ["understand_request", "generate_response"]
    assert state.results[0].model == "fake-decision/typed-decisions"
    assert decision_provider.calls == 1


def test_orchestrator_stops_after_retry_limit() -> None:
    provider = FakeProvider(failures_before_success=5)
    orchestrator = build_orchestrator(provider, max_retries=1)

    state = orchestrator.run("This will fail")

    assert state.status is TaskStatus.FAILED
    assert state.retry_count == 1
    assert provider.calls == 2
    assert state.final_output is None


def test_orchestrator_supports_cancellation_before_inference() -> None:
    provider = FakeProvider()
    orchestrator = build_orchestrator(provider)

    state = orchestrator.run("Cancel this", is_cancelled=lambda: True)

    assert state.status is TaskStatus.CANCELLED
    assert state.cancellation_requested is True
    assert provider.calls == 0


def test_orchestrator_honors_cancellation_after_inference() -> None:
    provider = FakeProvider()
    orchestrator = build_orchestrator(provider)
    cancelled = False
    original_chat = provider.chat

    def chat_then_cancel(request: ChatRequest) -> ChatResponse:
        nonlocal cancelled
        response = original_chat(request)
        cancelled = True
        return response

    provider.chat = chat_then_cancel

    state = orchestrator.run("Cancel while generating", is_cancelled=lambda: cancelled)

    assert state.status is TaskStatus.CANCELLED
    assert state.final_output is None
    assert state.cancellation_requested


def test_cancellation_during_provider_error_stops_retries() -> None:
    provider = FakeProvider()
    orchestrator = build_orchestrator(provider, max_retries=3)
    cancelled = False

    def failing_chat(request: ChatRequest) -> ChatResponse:
        nonlocal cancelled
        provider.calls += 1
        cancelled = True
        raise ProviderError("provider stopped", provider="fake", retryable=True)

    provider.chat = failing_chat

    state = orchestrator.run("Cancel while retrying", is_cancelled=lambda: cancelled)

    assert state.status is TaskStatus.CANCELLED
    assert state.retry_count == 0
    assert provider.calls == 1


def test_orchestrator_executes_explicit_tool_and_continues() -> None:
    provider = FakeProvider(issue_tool_call=True)
    executor = ToolExecutor(build_builtin_registry())
    orchestrator = build_orchestrator(provider, tool_executor=executor)

    state = orchestrator.run("Calculate 2 + 2", tool_names=("calculator",))

    assert state.status is TaskStatus.COMPLETED
    assert state.final_output == "completed locally"
    assert [result.step for result in state.results] == ["tool:calculator", "generate_response"]
    assert state.results[0].output == '{"value": 4}'
    assert provider.calls == 2


def test_orchestrator_pauses_for_confirmation() -> None:
    provider = FakeProvider(issue_tool_call=True)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculator",
            description="A confirmation test tool.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=lambda expression: "done",
            permission=PermissionClass.CONFIRMATION_REQUIRED,
        )
    )
    orchestrator = build_orchestrator(provider, tool_executor=ToolExecutor(registry))

    state = orchestrator.run("Run the confirmation tool", tool_names=("calculator",))

    assert state.status is TaskStatus.WAITING_FOR_CONFIRMATION
    assert state.current_step == "confirm:calculator"
    assert provider.calls == 1
    assert state.pending_action is not None
    assert state.pending_action.name == "calculator"
    assert state.pending_action.arguments == {"expression": "2 + 2"}


def build_confirmation_orchestrator(provider, handler):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="dangerous_action",
            description="A test action that requires approval.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler=handler,
            permission=PermissionClass.CONFIRMATION_REQUIRED,
        )
    )
    model_registry = ModelRegistry()
    model_registry.register(
        ModelInfo(
            model_id="local-test-model",
            provider="fake",
            local=True,
            available=True,
            capabilities=frozenset({"completion", "tools"}),
        )
    )
    store = InMemoryTaskStateStore()
    return Orchestrator(
        ModelGateway([provider]),
        DeterministicModelRouter(model_registry),
        store=store,
        provider_settings=Settings(local_only=True),
        tool_executor=ToolExecutor(registry),
    ), store


def test_approved_confirmation_executes_once_and_completes() -> None:
    provider = FakeProvider()
    responses = iter((
        ChatResponse(
            model="local-test-model",
            content="",
            tool_calls=(ToolCall(name="dangerous_action", arguments={"value": "approved"}),),
        ),
        ChatResponse(model="local-test-model", content="The action is done"),
    ))
    provider.chat = lambda request: next(responses)
    actions = []
    orchestrator, store = build_confirmation_orchestrator(provider, lambda value: actions.append(value) or {"done": True})

    paused = orchestrator.run("Do the approved action", tool_names=("dangerous_action",))
    assert paused.status is TaskStatus.WAITING_FOR_CONFIRMATION
    assert paused.pending_action is not None

    approved = store.resolve_confirmation(
        paused.task_id, paused.pending_action.confirmation_token, approved=True
    )
    completed = orchestrator.resume_confirmation(approved.task_id)

    assert approved.status is TaskStatus.PLANNING
    assert completed.status is TaskStatus.COMPLETED, completed.errors
    assert completed.final_output == "The action is done"
    assert actions == ["approved"]
    assert completed.pending_action is None


def test_rejected_confirmation_never_runs_tool() -> None:
    provider = FakeProvider()
    provider.chat = lambda request: ChatResponse(
        model=request.model,
        content="",
        tool_calls=(ToolCall(name="dangerous_action", arguments={"value": "no"}),),
    )
    actions = []
    orchestrator, store = build_confirmation_orchestrator(provider, lambda value: actions.append(value))

    paused = orchestrator.run("Do the action", tool_names=("dangerous_action",))
    denied = store.resolve_confirmation(
        paused.task_id, paused.pending_action.confirmation_token, approved=False
    )

    assert denied.status is TaskStatus.FAILED
    assert denied.pending_action is None
    assert actions == []


def test_task_state_rejects_invalid_transition_and_unknown_save() -> None:
    store = InMemoryTaskStateStore()
    state = store.create("A task")

    with pytest.raises(TaskStateError):
        state.transition(TaskStatus.COMPLETED)
    with pytest.raises(TaskStateError):
        store.save(state.with_updates(task_id="missing"))
