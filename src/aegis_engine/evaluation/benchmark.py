"""Small, provider-neutral benchmark runner for local model comparison."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter

from aegis_engine.models import ChatMessage, ChatRequest, ModelGateway, ModelInfo, ProviderError
from aegis_engine.verification import classify_error


ScoreFunction = Callable[[str], float]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prompt: str
    score: ScoreFunction | None = None
    allowed_tools: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.prompt.strip():
            raise ValueError("evaluation case_id and prompt are required")


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    model: str
    success: bool
    latency_ms: float
    quality_score: float | None = None
    tool_calls_valid: bool = True
    output: str = ""
    error_category: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    results: tuple[EvaluationResult, ...]

    @property
    def success_rate(self) -> float:
        return sum(result.success for result in self.results) / len(self.results) if self.results else 0.0

    @property
    def average_latency_ms(self) -> float:
        return sum(result.latency_ms for result in self.results) / len(self.results) if self.results else 0.0


class BenchmarkRunner:
    """Run each case against each available model through the gateway."""

    def __init__(self, gateway: ModelGateway, models: Iterable[ModelInfo]) -> None:
        self.gateway = gateway
        self.models = tuple(models)

    def run(self, cases: Iterable[EvaluationCase], *, max_tokens: int = 512) -> EvaluationSummary:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        benchmark_cases = tuple(cases)
        results: list[EvaluationResult] = []
        for model in self.models:
            for case in benchmark_cases:
                started = perf_counter()
                try:
                    response = self.gateway.chat(
                        model,
                        ChatRequest(
                            model=model.model_id,
                            messages=(ChatMessage(role="user", content=case.prompt),),
                            max_tokens=max_tokens,
                        ),
                    )
                    tool_calls_valid = all(call.name in case.allowed_tools for call in response.tool_calls)
                    quality_score = case.score(response.content) if case.score else None
                    success = bool(response.content.strip()) and tool_calls_valid
                    results.append(
                        EvaluationResult(
                            case_id=case.case_id,
                            model=f"{model.provider}/{model.model_id}",
                            success=success,
                            latency_ms=(perf_counter() - started) * 1000,
                            quality_score=quality_score,
                            tool_calls_valid=tool_calls_valid,
                            output=response.content,
                            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                            completion_tokens=response.usage.completion_tokens if response.usage else None,
                        )
                    )
                except ProviderError as exc:
                    results.append(
                        EvaluationResult(
                            case_id=case.case_id,
                            model=f"{model.provider}/{model.model_id}",
                            success=False,
                            latency_ms=(perf_counter() - started) * 1000,
                            error_category=classify_error(exc).value,
                            error=str(exc),
                        )
                    )
        return EvaluationSummary(results=tuple(results))
