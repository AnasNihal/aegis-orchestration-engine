from dataclasses import dataclass

import pytest

from mmo_engine.config import Settings
from mmo_engine.models import (
    ChatRequest,
    ChatResponse,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
    ProviderError,
)
from mmo_engine.orchestration import Orchestrator, OrchestratorConfig
from mmo_engine.tasks import InMemoryTaskStateStore, TaskStateError, TaskStatus


@dataclass
class FakeProvider:
    name: str = "fake"
    calls: int = 0
    failures_before_success: int = 0

    def list_models(self) -> list[ModelInfo]:
        return []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ProviderError("temporary provider failure", provider=self.name, retryable=True)
        return ChatResponse(model=request.model, content="completed locally")


def build_orchestrator(provider: FakeProvider, *, max_retries: int = 1):
    registry = ModelRegistry()
    registry.register(
        ModelInfo(
            model_id="local-test-model",
            provider="fake",
            local=True,
            available=True,
            capabilities=frozenset({"completion"}),
        )
    )
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry)
    return Orchestrator(
        gateway,
        router,
        provider_settings=Settings(local_only=True),
        config=OrchestratorConfig(max_retries=max_retries),
    )


def test_orchestrator_completes_and_records_selected_model() -> None:
    provider = FakeProvider()
    orchestrator = build_orchestrator(provider)

    state = orchestrator.run("Summarize this task")

    assert state.status is TaskStatus.COMPLETED
    assert state.final_output == "completed locally"
    assert state.selected_models == ("fake/local-test-model",)
    assert state.completed_steps == ("generate_response",)
    assert provider.calls == 1


def test_orchestrator_retries_only_bounded_retryable_failures() -> None:
    provider = FakeProvider(failures_before_success=1)
    orchestrator = build_orchestrator(provider, max_retries=1)

    state = orchestrator.run("Retry this task")

    assert state.status is TaskStatus.COMPLETED
    assert state.retry_count == 1
    assert provider.calls == 2


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


def test_task_state_rejects_invalid_transition_and_unknown_save() -> None:
    store = InMemoryTaskStateStore()
    state = store.create("A task")

    with pytest.raises(TaskStateError):
        state.transition(TaskStatus.COMPLETED)
    with pytest.raises(TaskStateError):
        store.save(state.with_updates(task_id="missing"))
