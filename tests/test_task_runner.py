from threading import Event
from time import monotonic, sleep

import pytest

from aegis_engine.tasks import InMemoryTaskStateStore, TaskStatus
from aegis_engine.tasks.runner import TaskCapacityError, TaskRunner


class BlockingOrchestrator:
    def __init__(self) -> None:
        self.store = InMemoryTaskStateStore()
        self.started = Event()
        self.release = Event()
        self.finished = Event()

    def run(self, request, *, task_id, is_cancelled, **kwargs):
        state = self.store.claim(task_id)
        self.started.set()
        self.release.wait(timeout=3)
        if is_cancelled():
            state = state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED)
        else:
            state = state.transition(TaskStatus.EXECUTING).with_updates(
                final_output="finished"
            ).transition(TaskStatus.COMPLETED)
        self.store.save(state)
        self.finished.set()


class FailingOrchestrator:
    def __init__(self) -> None:
        self.store = InMemoryTaskStateStore()

    def run(self, request, *, task_id, **kwargs):
        self.store.claim(task_id)
        raise RuntimeError("secret provider detail")


def wait_for_status(store, task_id: str, status: TaskStatus) -> None:
    deadline = monotonic() + 2
    while monotonic() < deadline:
        if store.get(task_id).status is status:
            return
        sleep(0.01)
    pytest.fail(f"task did not reach {status}")


def test_runner_bounds_capacity_and_persists_cancellation() -> None:
    orchestrator = BlockingOrchestrator()
    runner = TaskRunner(orchestrator, max_workers=1, max_queued=0)
    try:
        task = runner.submit("first", model_id="test")
        assert orchestrator.started.wait(timeout=2)
        with pytest.raises(TaskCapacityError, match="capacity reached"):
            runner.submit("second", model_id="test")

        cancellation = runner.cancel(task.task_id)
        assert cancellation.cancellation_requested
        orchestrator.release.set()
        assert orchestrator.finished.wait(timeout=2)
        assert orchestrator.store.get(task.task_id).status is TaskStatus.CANCELLED
    finally:
        orchestrator.release.set()
        runner.close()


def test_runner_records_sanitized_failure_on_unexpected_exception() -> None:
    orchestrator = FailingOrchestrator()
    runner = TaskRunner(orchestrator, max_workers=1, max_queued=0)
    try:
        task = runner.submit("fail", model_id="test")
        wait_for_status(orchestrator.store, task.task_id, TaskStatus.FAILED)
        state = orchestrator.store.get(task.task_id)
        assert state.status is TaskStatus.FAILED
        assert state.errors[-1] == "Background task failed unexpectedly"
        assert "secret" not in str(state)
    finally:
        runner.close()
