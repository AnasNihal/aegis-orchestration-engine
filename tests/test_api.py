from dataclasses import dataclass

from fastapi.testclient import TestClient

from aegis_engine.api import create_app
from aegis_engine.models import ChatRequest, ChatResponse, ModelInfo
from aegis_engine.tasks import TaskStatus


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
