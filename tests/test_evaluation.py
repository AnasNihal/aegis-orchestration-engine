from dataclasses import dataclass

from aegis_engine.evaluation import BenchmarkRunner, EvaluationCase
from aegis_engine.models import (
    ChatRequest,
    ChatResponse,
    DeterministicModelRouter,
    ModelGateway,
    ModelInfo,
    ModelRegistry,
    UsageMetadata,
)


@dataclass
class FakeProvider:
    name: str = "fake"

    def list_models(self) -> list[ModelInfo]:
        return []

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            model=request.model,
            content="4",
            usage=UsageMetadata(prompt_tokens=3, completion_tokens=1),
        )


def test_benchmark_tracks_success_latency_quality_and_usage() -> None:
    model = ModelInfo(
        model_id="model-a",
        provider="fake",
        local=True,
        available=True,
        capabilities=frozenset({"completion"}),
    )
    registry = ModelRegistry()
    registry.register(model)
    runner = BenchmarkRunner(ModelGateway([FakeProvider()]), [model])

    summary = runner.run([EvaluationCase("math", "2 + 2", score=lambda output: 1.0 if output == "4" else 0.0)])

    assert summary.success_rate == 1.0
    assert summary.results[0].quality_score == 1.0
    assert summary.results[0].prompt_tokens == 3
    assert summary.results[0].completion_tokens == 1
