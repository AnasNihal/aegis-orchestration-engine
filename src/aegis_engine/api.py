"""FastAPI interface for the Aegis orchestration engine.

The API layer owns HTTP validation and transport. It does not contain model
selection or tool policy; those remain in the provider-neutral core.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Callable, Mapping
import json
from queue import Queue
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from aegis_engine.config import Settings, settings
from aegis_engine.models import (
    ChatMessage,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
    OllamaProvider,
    ProviderError,
)
from aegis_engine.orchestration import Orchestrator
from aegis_engine.storage import SQLiteTaskStateStore
from aegis_engine.tasks import TaskStateError, TaskStatus
from aegis_engine.tasks.runner import TaskCapacityError, TaskRunner
from aegis_engine.tools import ToolExecutor, build_builtin_registry, select_tools
from aegis_engine.web import INDEX_HTML


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=32_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


class ModelResponse(BaseModel):
    model_id: str
    provider: str
    capabilities: list[str]
    local: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ErrorResponse(BaseModel):
    error: str


class TaskResultResponse(BaseModel):
    step: str
    success: bool
    output: str
    model: str | None = None
    error: str | None = None


class TaskStepResponse(BaseModel):
    name: str
    status: str
    error: str | None = None


class PendingActionResponse(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    confirmation_token: str


class TaskResponse(BaseModel):
    task_id: str
    user_request: str
    status: str
    phase: str
    plan: list[str]
    steps: list[TaskStepResponse]
    current_step: str | None
    completed_steps: list[str]
    results: list[TaskResultResponse]
    errors: list[str]
    retry_count: int
    selected_models: list[str]
    final_output: str | None
    cancellation_requested: bool
    pending_action: PendingActionResponse | None
    created_at: str
    updated_at: str


class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: Literal["pending"]


class TaskCancellationResponse(BaseModel):
    task_id: str
    status: str
    cancellation_requested: bool


class TaskConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_token: str = Field(min_length=32, max_length=128)
    approved: bool


class TaskConfirmationResponse(BaseModel):
    task_id: str
    status: str


def build_runtime(config: Settings) -> tuple[Orchestrator, tuple[ModelInfo, ...]]:
    """Discover models and build the shared API runtime."""

    provider = OllamaProvider(config)
    registry = ModelRegistry()
    registry.refresh(provider)
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry, preferred_model=config.default_model)
    orchestrator = Orchestrator(
        gateway,
        router,
        provider_settings=config,
        store=SQLiteTaskStateStore(config.task_db_path),
        tool_executor=ToolExecutor(build_builtin_registry(config.approved_file_roots)),
    )
    models = tuple(model for model in registry.available() if "completion" in model.capabilities)
    return orchestrator, models


def create_app(
    config: Settings | None = None,
    *,
    orchestrator: Orchestrator | None = None,
    models: tuple[ModelInfo, ...] | None = None,
) -> FastAPI:
    """Create a validated FastAPI application.

    Tests and embedding applications can inject an orchestrator and model
    catalog. Normal CLI startup discovers the configured Ollama models.
    """

    runtime_config = config or settings
    if orchestrator is None:
        orchestrator, discovered_models = build_runtime(runtime_config)
        models = discovered_models
    available_models = tuple(models or ())
    model_ids = {model.model_id for model in available_models}
    runner = (
        TaskRunner(
            orchestrator,
            max_workers=runtime_config.task_max_workers,
            max_queued=runtime_config.task_max_queued,
        )
        if hasattr(orchestrator, "store") else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if runner is not None:
                await asyncio.to_thread(runner.close)

    app = FastAPI(
        title="Aegis Orchestration Engine",
        version="0.1.0",
        description="Local-first, model-agnostic AI orchestration API.",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/models", response_model=list[ModelResponse], tags=["models"])
    async def list_models() -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id=model.model_id,
                provider=model.provider,
                capabilities=sorted(model.capabilities),
                local=model.local,
            )
            for model in available_models
        ]

    @app.get("/api/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
    async def get_task(task_id: str) -> TaskResponse:
        try:
            task = orchestrator.store.get(task_id)
        except TaskStateError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return TaskResponse(
            task_id=task.task_id,
            user_request=task.user_request,
            status=task.status.value,
            phase=task.phase,
            plan=list(task.plan),
            steps=[
                TaskStepResponse(name=step.name, status=step.status.value, error=step.error)
                for step in task.steps
            ],
            current_step=task.current_step,
            completed_steps=list(task.completed_steps),
            results=[TaskResultResponse(**result.__dict__) for result in task.results],
            errors=list(task.errors),
            retry_count=task.retry_count,
            selected_models=list(task.selected_models),
            final_output=task.final_output,
            cancellation_requested=task.cancellation_requested,
            pending_action=(
                PendingActionResponse(
                    tool_name=task.pending_action.name,
                    arguments=task.pending_action.arguments,
                    confirmation_token=task.pending_action.confirmation_token,
                )
                if task.status is TaskStatus.WAITING_FOR_CONFIRMATION and task.pending_action
                else None
            ),
            created_at=task.created_at,
            updated_at=task.updated_at,
        )

    @app.post(
        "/api/tasks",
        response_model=TaskAcceptedResponse,
        status_code=202,
        tags=["tasks"],
    )
    async def submit_task(body: ChatRequestBody) -> TaskAcceptedResponse:
        if model_ids and body.model not in model_ids:
            raise HTTPException(status_code=422, detail="selected model is not available")
        if runner is None:
            raise HTTPException(status_code=503, detail="task runner is unavailable")
        conversation = tuple(ChatMessage(role=item.role, content=item.content) for item in body.history)
        try:
            task = runner.submit(
                body.message,
                model_id=body.model,
                conversation=conversation,
                tool_names=select_tools(body.message),
            )
        except TaskCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return TaskAcceptedResponse(task_id=task.task_id, status="pending")

    @app.post(
        "/api/tasks/{task_id}/cancel",
        response_model=TaskCancellationResponse,
        tags=["tasks"],
    )
    async def cancel_task(task_id: str) -> TaskCancellationResponse:
        if runner is None:
            raise HTTPException(status_code=503, detail="task runner is unavailable")
        try:
            task = runner.cancel(task_id)
        except TaskStateError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return TaskCancellationResponse(
            task_id=task_id,
            status=task.status.value,
            cancellation_requested=task.cancellation_requested,
        )

    @app.post(
        "/api/tasks/{task_id}/confirm",
        response_model=TaskConfirmationResponse,
        tags=["tasks"],
    )
    async def confirm_task(task_id: str, body: TaskConfirmationRequest) -> TaskConfirmationResponse:
        if runner is None:
            raise HTTPException(status_code=503, detail="task runner is unavailable")
        try:
            task = runner.confirm(
                task_id,
                body.confirmation_token,
                approved=body.approved,
            )
        except TaskCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except TaskStateError as exc:
            code = 403 if "confirmation token" in str(exc) else 409
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        return TaskConfirmationResponse(task_id=task_id, status=task.status.value)

    @app.post(
        "/api/chat",
        response_class=StreamingResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["chat"],
    )
    async def chat(body: ChatRequestBody) -> StreamingResponse:
        if model_ids and body.model not in model_ids:
            raise HTTPException(status_code=422, detail="selected model is not available")
        conversation = tuple(ChatMessage(role=item.role, content=item.content) for item in body.history)

        async def event_stream():
            events: Queue[dict[str, Any] | None] = Queue()

            def emit_token(token: str) -> None:
                events.put({"type": "token", "content": token})

            def run_task() -> None:
                try:
                    task = orchestrator.run(
                        body.message,
                        model_id=body.model,
                        conversation=conversation,
                        tool_names=select_tools(body.message),
                        on_token=emit_token,
                    )
                    if task.status is not TaskStatus.COMPLETED:
                        events.put({"type": "error", "error": task.errors[-1] if task.errors else str(task.status)})
                    else:
                        events.put(
                            {
                                "type": "complete",
                                "task_id": task.task_id,
                                "status": task.status,
                                "models": list(task.selected_models),
                            }
                        )
                except (ProviderError, OSError, ValueError) as exc:
                    events.put({"type": "error", "error": str(exc)})
                finally:
                    events.put(None)

            task = asyncio.create_task(asyncio.to_thread(run_task))
            try:
                while True:
                    event = await asyncio.to_thread(events.get)
                    if event is None:
                        break
                    yield json.dumps(event, default=str) + "\n"
            finally:
                await task

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
