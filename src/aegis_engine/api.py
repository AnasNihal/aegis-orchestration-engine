"""FastAPI interface for the Aegis orchestration engine.

The API layer owns HTTP validation and transport. It does not contain model
selection or tool policy; those remain in the provider-neutral core.
"""

from __future__ import annotations

import asyncio
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
from aegis_engine.tasks import TaskStatus
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
    app = FastAPI(
        title="Aegis Orchestration Engine",
        version="0.1.0",
        description="Local-first, model-agnostic AI orchestration API.",
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
