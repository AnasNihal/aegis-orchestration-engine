from dataclasses import dataclass

import pytest

from aegis_engine.decisions import (
    DecisionRequest,
    DecisionResponse,
    LayaDecisionEngine,
    LayaTaskUnderstanding,
    LayaUnavailableError,
)


@dataclass
class FakeLayaRouter:
    calls: int = 0

    def predict(self, state, questions, **kwargs):
        self.calls += 1
        return {
            "domain": {"label": "code", "confidence": 0.91},
            "needs_tools": {"probability": 0.12},
            "routing": {"model": kwargs.get("model", "typed-decisions")},
        }


def test_laya_adapter_normalizes_typed_decisions_without_importing_package() -> None:
    router = FakeLayaRouter()
    provider = LayaDecisionEngine(router=router)

    response = provider.predict(
        DecisionRequest(
            state={"request": "Explain this Python function"},
            questions={"domain": {"type": "choice", "criteria": {"code": "code"}}},
        )
    )

    assert response.provider == "laya"
    assert response.model == "typed-decisions"
    assert response.decisions["domain"]["label"] == "code"
    assert response.latency_ms is not None
    assert router.calls == 1


def test_task_understanding_uses_laya_typed_decisions_workflow() -> None:
    router = FakeLayaRouter()
    understanding = LayaTaskUnderstanding(LayaDecisionEngine(router=router))

    response = understanding.analyze("Read a CSV and calculate the average")

    assert response.model == "typed-decisions"
    assert set(response.decisions) >= {"domain", "needs_tools"}


def test_laya_adapter_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    provider = LayaDecisionEngine()

    with pytest.raises(LayaUnavailableError, match="uv sync --extra laya"):
        provider.predict(
            DecisionRequest(
                state={"request": "hello"},
                questions={"domain": {"type": "choice", "criteria": {"other": "other"}}},
            )
        )
