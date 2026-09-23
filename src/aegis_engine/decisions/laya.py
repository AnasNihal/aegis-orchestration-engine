"""Optional adapter for the Laya typed-decision engine.

Laya is intentionally not implemented as a chat ``ModelProvider``. It makes
fast typed decisions, while Ollama remains responsible for natural-language
generation in Aegis.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from .base import DecisionProviderError, DecisionRequest, DecisionResponse


class LayaUnavailableError(DecisionProviderError):
    """Raised when the optional Laya package is not installed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, provider="laya", retryable=False)


class LayaDecisionEngine:
    """Lazy, optional adapter around ``laya.Router``.

    The Laya package and its Hugging Face checkpoint are loaded only when the
    first prediction is requested. ``preload`` is opt-in because preloading
    downloads and builds checkpoints eagerly.
    """

    name = "laya"

    def __init__(
        self,
        *,
        router: Any | None = None,
        model: str | None = None,
        device: str | None = None,
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = True,
        preload: bool = False,
    ) -> None:
        if max_loaded < 1:
            raise ValueError("max_loaded must be positive")
        self._router = router
        self.model = model
        self.device = device
        self.max_loaded = max_loaded
        self.default = default
        self.auto_task_detection = auto_task_detection
        self.preload = preload

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        router = self._get_router()
        started = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if request.model or self.model:
            kwargs["model"] = request.model or self.model
        try:
            result = router.predict(request.state, dict(request.questions), **kwargs)
        except DecisionProviderError:
            raise
        except Exception as exc:
            raise DecisionProviderError(
                f"Laya prediction failed: {exc}", provider=self.name, retryable=False
            ) from exc
        if not isinstance(result, Mapping):
            raise DecisionProviderError(
                "Laya returned a non-mapping result", provider=self.name, retryable=False
            )

        routing = result.get("routing", {})
        selected_model = self.model or "auto"
        if isinstance(routing, Mapping) and routing.get("model"):
            selected_model = str(routing["model"])
        return DecisionResponse(
            provider=self.name,
            model=selected_model,
            decisions=dict(result),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=dict(result),
        )

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        try:
            import laya
        except ImportError as exc:
            raise LayaUnavailableError(
                "Laya is not installed; run `uv sync --extra laya` to enable it"
            ) from exc
        self._router = laya.Router(
            device=self.device,
            max_loaded=self.max_loaded,
            default=self.default,
            auto_task_detection=self.auto_task_detection,
            preload=False,
        )
        if self.preload:
            # Preload only the explicitly configured checkpoint. The normal
            # Aegis task-understanding workflow requests typed-decisions and
            # should not resident-load every Laya checkpoint.
            self._router.preload([self.model or "typed-decisions"])
        return self._router
