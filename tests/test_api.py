from dataclasses import dataclass
from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient

from aegis_engine.api import create_app
from aegis_engine.config import Settings
from aegis_engine.models import ChatRequest, ChatResponse, ModelInfo
from aegis_engine.tasks import InMemoryTaskStateStore, PendingToolAction, TaskStatus


@dataclass
class FakeTask:
    task_id: str = "task-api"
    status: TaskStatus = TaskStatus.COMPLETED
    errors: tuple[str, ...] = ()
    selected_models: tuple[str, ...] = ("fake/model",)
    final_output: str = "hello from api"


class FakeOrchestrator:
    def run(self, request: str, **kwargs) -> FakeTask:
        callback = kwargs.get("on_token")
        if callback:
            callback("hello ")
            callback("from api")
        return FakeTask()


class FakeStore:
    def __init__(self, task):
        self.task = task

    def get(self, task_id):
        return self.task


class AsyncFakeOrchestrator:
    def __init__(self) -> None:
        self.store = InMemoryTaskStateStore()

    def run(self, request: str, *, task_id: str, is_cancelled, **kwargs):
        task = self.store.get(task_id)
        if is_cancelled():
            return self.store.save(
                task.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED)
            )
        return self.store.save(
            task.transition(TaskStatus.PLANNING).transition(TaskStatus.EXECUTING).with_updates(
                final_output="background complete"
            ).transition(TaskStatus.COMPLETED)
        )


class SlowOrchestrator:
    def __init__(self) -> None:
        self.store = InMemoryTaskStateStore()
        self.started = Event()
        self.release = Event()

    def run(self, request: str, *, task_id: str, is_cancelled, **kwargs):
        state = self.store.claim(task_id)
        self.started.set()
        self.release.wait(timeout=3)
        if is_cancelled():
            state = state.with_updates(cancellation_requested=True).transition(TaskStatus.CANCELLED)
        else:
            state = state.transition(TaskStatus.EXECUTING).transition(TaskStatus.COMPLETED)
        return self.store.save(state)


def test_fastapi_exposes_typed_health_models_and_openapi() -> None:
    model = ModelInfo(
        model_id="fake-model",
        provider="fake",
        local=True,
        available=True,
        capabilities=frozenset({"completion"}),
    )
    client = TestClient(create_app(orchestrator=FakeOrchestrator(), models=(model,)))

    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/models").json()[0]["model_id"] == "fake-model"
    assert client.get("/openapi.json").status_code == 200


def test_fastapi_validates_chat_and_streams_ndjson() -> None:
    model = ModelInfo(
        model_id="fake-model",
        provider="fake",
        local=True,
        available=True,
        capabilities=frozenset({"completion"}),
    )
    client = TestClient(create_app(orchestrator=FakeOrchestrator(), models=(model,)))

    response = client.post("/api/chat", json={"model": "fake-model", "message": "hello"})

    assert response.status_code == 200
    assert '"type": "token"' in response.text
    assert '"type": "complete"' in response.text


def test_fastapi_rejects_unknown_model() -> None:
    model = ModelInfo(
        model_id="fake-model",
        provider="fake",
        local=True,
        available=True,
        capabilities=frozenset({"completion"}),
    )
    client = TestClient(create_app(orchestrator=FakeOrchestrator(), models=(model,)))

    response = client.post("/api/chat", json={"model": "missing", "message": "hello"})

    assert response.status_code == 422


def test_fastapi_exposes_persisted_task_state() -> None:
    from aegis_engine.tasks import TaskResult, TaskState

    task = TaskState(task_id="task-api", user_request="hello").transition(TaskStatus.PLANNING)
    task = task.with_updates(
        plan=("generate_response",),
        results=(TaskResult(step="generate_response", success=True, output="done"),),
    )
    orchestrator = FakeOrchestrator()
    orchestrator.store = FakeStore(task)
    client = TestClient(create_app(orchestrator=orchestrator, models=()))

    response = client.get("/api/tasks/task-api")

    assert response.status_code == 200
    assert response.json()["plan"] == ["generate_response"]
    assert response.json()["steps"] == []
    assert response.json()["results"][0]["output"] == "done"


def test_fastapi_submits_background_task_and_allows_status_polling() -> None:
    orchestrator = AsyncFakeOrchestrator()
    with TestClient(create_app(orchestrator=orchestrator, models=())) as client:
        accepted = client.post("/api/tasks", json={"model": "fake-model", "message": "run this task"})

        assert accepted.status_code == 202
        task_id = accepted.json()["task_id"]
        deadline = monotonic() + 2
        while monotonic() < deadline:
            status = client.get(f"/api/tasks/{task_id}")
            if status.json()["status"] == "completed":
                break
            sleep(0.01)
        assert status.status_code == 200
        assert status.json()["status"] == "completed"
        assert status.json()["final_output"] == "background complete"


def test_api_startup_recovers_interrupted_sqlite_task(tmp_path) -> None:
    from aegis_engine.storage import SQLiteTaskStateStore

    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStateStore(path)
    store.claim(store.create("interrupted", task_id="old-task").task_id)
    orchestrator = AsyncFakeOrchestrator()
    orchestrator.store = SQLiteTaskStateStore(path)

    with TestClient(create_app(
        config=Settings(task_db_path=str(path)), orchestrator=orchestrator, models=()
    )) as client:
        response = client.get("/api/tasks/old-task")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "server restart" in response.json()["errors"][-1]


def test_task_api_rejects_full_queue_and_persists_cancel_request() -> None:
    orchestrator = SlowOrchestrator()
    app = create_app(
        config=Settings(task_max_workers=1, task_max_queued=0),
        orchestrator=orchestrator,
        models=(),
    )
    try:
        with TestClient(app) as client:
            first = client.post("/api/tasks", json={"model": "fake", "message": "first"})
            assert first.status_code == 202
            assert orchestrator.started.wait(timeout=2)

            second = client.post("/api/tasks", json={"model": "fake", "message": "second"})
            assert second.status_code == 503

            task_id = first.json()["task_id"]
            cancelled = client.post(f"/api/tasks/{task_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["cancellation_requested"] is True
            orchestrator.release.set()
            deadline = monotonic() + 2
            while monotonic() < deadline:
                status = client.get(f"/api/tasks/{task_id}").json()["status"]
                if status == "cancelled":
                    break
                sleep(0.01)
            assert status == "cancelled"
    finally:
        orchestrator.release.set()


def test_task_confirmation_endpoint_checks_token_and_resumes_task() -> None:
    class ConfirmableOrchestrator:
        def __init__(self) -> None:
            self.store = InMemoryTaskStateStore()
            state = self.store.create("run approved action", task_id="confirm-api")
            state = state.transition(TaskStatus.PLANNING).with_updates(
                pending_action=PendingToolAction(
                    name="example_action",
                    arguments={"value": "safe"},
                    call_id="call-1",
                    model_id="local-model",
                    allowed_tools=("example_action",),
                    messages=(),
                    confirmation_token="api-confirmation-token-0123456789",
                )
            ).transition(TaskStatus.WAITING_FOR_CONFIRMATION)
            self.store.save(state)

        def run(self, request: str, **kwargs):
            raise AssertionError("the seeded confirmation task should not be resubmitted")

        def resume_confirmation(self, task_id: str, **kwargs):
            state = self.store.get(task_id)
            state = self.store.save(
                state.transition(TaskStatus.EXECUTING).with_updates(final_output="done")
            )
            return self.store.save(state.transition(TaskStatus.COMPLETED))

    orchestrator = ConfirmableOrchestrator()
    with TestClient(create_app(orchestrator=orchestrator, models=())) as client:
        wrong = client.post(
            "/api/tasks/confirm-api/confirm",
            json={"confirmation_token": "incorrect-confirmation-token-0000", "approved": True},
        )
        assert wrong.status_code == 403

        approved = client.post(
            "/api/tasks/confirm-api/confirm",
            json={"confirmation_token": "api-confirmation-token-0123456789", "approved": True},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "planning"

        deadline = monotonic() + 2
        while monotonic() < deadline:
            task = client.get("/api/tasks/confirm-api").json()
            if task["status"] == "completed":
                break
            sleep(0.01)
        assert task["status"] == "completed"
        assert task["final_output"] == "done"
