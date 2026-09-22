import json
from typing import Any
from urllib.error import URLError

import pytest

from aegis_engine.config import Settings
from aegis_engine.models import ChatMessage, ChatRequest, ModelProvider, OllamaProvider, ProviderError


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def fake_opener(payload: dict[str, Any], status: int = 200):
    captured: list[Any] = []

    def opener(request, timeout):
        captured.append((request, timeout))
        return FakeResponse(payload, status)

    opener.captured = captured
    return opener


def test_list_models_discovers_models_without_inventing_capabilities() -> None:
    opener = fake_opener(
        {
            "models": [
                {"name": "qwen2.5:7b", "size": 123, "digest": "abc"},
                {"name": "nomic-embed-text:latest", "size": 456},
            ]
        }
    )
    provider = OllamaProvider(Settings(), opener=opener)

    models = provider.list_models()

    assert [model.model_id for model in models] == [
        "qwen2.5:7b",
        "nomic-embed-text:latest",
    ]
    assert all(model.local and model.available for model in models)
    assert all(not model.capabilities for model in models)
    assert opener.captured[0][0].full_url.endswith("/api/tags")


def test_list_models_uses_capabilities_reported_by_ollama() -> None:
    captured: list[Any] = []

    def opener(request, timeout):
        captured.append((request, timeout))
        if request.full_url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": "qwen2.5:7b"}]})
        return FakeResponse({"capabilities": ["completion", "tools"], "details": {"family": "qwen2"}})

    models = OllamaProvider(Settings(), opener=opener).list_models()

    assert models[0].capabilities == frozenset({"completion", "tools"})
    assert models[0].metadata["details"] == {"family": "qwen2"}
    assert [request.full_url for request, _ in captured] == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/show",
    ]


def test_ollama_adapter_implements_provider_contract() -> None:
    provider = OllamaProvider(Settings(), opener=fake_opener({"models": []}))

    assert isinstance(provider, ModelProvider)


def test_chat_converts_request_and_tool_calls() -> None:
    opener = fake_opener(
        {
            "model": "qwen2.5:7b",
            "message": {
                "role": "assistant",
                "content": "The answer is ready.",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "calculator", "arguments": {"x": 2}}}
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 4,
        }
    )
    provider = OllamaProvider(Settings(request_timeout_seconds=7), opener=opener)
    request = ChatRequest(
        model="qwen2.5:7b",
        messages=(ChatMessage(role="user", content="Calculate 2."),),
        max_tokens=50,
        tools=({"type": "function", "function": {"name": "calculator"}},),
    )

    response = provider.chat(request)
    sent_request, timeout = opener.captured[0]
    sent_payload = json.loads(sent_request.data.decode("utf-8"))

    assert timeout == 7
    assert sent_payload["model"] == "qwen2.5:7b"
    assert sent_payload["messages"] == [{"role": "user", "content": "Calculate 2."}]
    assert sent_payload["options"]["num_predict"] == 50
    assert response.content == "The answer is ready."
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"x": 2}
    assert response.usage is not None
    assert response.usage.total_duration_ns is None


def test_provider_surfaces_http_failures_as_provider_errors() -> None:
    opener = fake_opener({"error": "temporarily unavailable"}, status=503)
    provider = OllamaProvider(Settings(), opener=opener)

    with pytest.raises(ProviderError) as error:
        provider.list_models()

    assert error.value.provider == "ollama"
    assert error.value.retryable is True


def test_provider_unavailable_is_normalized() -> None:
    def unavailable_opener(_request, timeout):
        assert timeout == 60
        raise URLError("connection refused")

    provider = OllamaProvider(Settings(), opener=unavailable_opener)

    with pytest.raises(ProviderError, match="Ollama is unavailable") as error:
        provider.list_models()

    assert error.value.retryable is True
