"""Local Ollama provider adapter.

This module owns Ollama's HTTP payloads and response conversion. The rest of
the engine depends only on the contracts in ``models.base``.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from mmo_engine.config import Settings, settings
from mmo_engine.models.base import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ProviderError,
    ToolCall,
    UsageMetadata,
)


class OllamaProvider:
    """A small, injectable adapter for Ollama's local HTTP API."""

    name = "ollama"

    def __init__(
        self,
        provider_settings: Settings | None = None,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.settings = provider_settings or settings
        self._opener = opener

    def _url(self, path: str) -> str:
        return urljoin(self.settings.ollama_base_url.rstrip("/") + "/", path.lstrip("/"))

    def _request(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path),
            data=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with self._opener(request, timeout=self.settings.request_timeout_seconds) as response:
                body = response.read()
                status = getattr(response, "status", 200)
            if status >= 400:
                raise ProviderError(
                    f"Ollama returned HTTP {status}", provider=self.name, retryable=status >= 500
                )
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ProviderError(
                    "Ollama returned a non-object JSON response", provider=self.name
                )
            return parsed
        except ProviderError:
            raise
        except HTTPError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.code}",
                provider=self.name,
                retryable=exc.code >= 500,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                "Ollama is unavailable; start the local Ollama service and try again",
                provider=self.name,
                retryable=True,
            ) from exc
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Ollama returned an invalid response", provider=self.name
            ) from exc

    def list_models(self) -> list[ModelInfo]:
        """Discover locally available models without assuming capabilities."""

        payload = self._request("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ProviderError("Ollama model list has an invalid shape", provider=self.name)
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            result.append(
                ModelInfo(
                    model_id=item["name"],
                    provider=self.name,
                    local=True,
                    available=True,
                    metadata={
                        key: item[key]
                        for key in ("digest", "size", "modified_at")
                        if key in item
                    },
                )
            )
        return result

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send one non-streaming chat request to Ollama."""

        if not request.model.strip():
            raise ProviderError("A model identifier is required", provider=self.name)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.as_dict() for message in request.messages],
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.tools:
            payload["tools"] = list(request.tools)

        response = self._request("/api/chat", payload=payload)
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
            raise ProviderError("Ollama chat response has no valid message", provider=self.name)

        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        if isinstance(raw_tool_calls, list):
            for item in raw_tool_calls:
                function = item.get("function") if isinstance(item, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = function.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        name=function["name"],
                        arguments=arguments,
                        call_id=item.get("id") if isinstance(item, dict) else None,
                    )
                )

        usage = UsageMetadata(
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            total_duration_ns=response.get("total_duration"),
        )
        return ChatResponse(
            model=str(response.get("model", request.model)),
            content=message["content"].strip(),
            tool_calls=tuple(tool_calls),
            usage=usage,
            raw=response,
        )

