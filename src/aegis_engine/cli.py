"""Command-line entry point for the local Aegis runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from collections.abc import Sequence

from aegis_engine.config import ConfigurationError, Settings
from aegis_engine.models import (
    DeterministicModelRouter,
    ModelGateway,
    ModelRegistry,
    OllamaProvider,
    ProviderError,
)
from aegis_engine.orchestration import Orchestrator
from aegis_engine.tasks import TaskStatus


def build_orchestrator(config: Settings) -> Orchestrator:
    """Discover local Ollama models and construct the bounded runtime."""

    provider = OllamaProvider(config)
    registry = ModelRegistry()
    registry.refresh(provider)
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry, preferred_model=config.default_model)
    return Orchestrator(gateway, router, provider_settings=config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Aegis orchestration task.")
    parser.add_argument("request", nargs="?", help="The task to execute.")
    parser.add_argument("--model", help="Prefer a specific installed Ollama model.")
    parser.add_argument(
        "--no-laya",
        action="store_true",
        help="Disable optional Laya task understanding for this request.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print selected models and non-sensitive task errors.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = args.request or input("Aegis> ").strip()
    if not request:
        print("A request is required.", file=sys.stderr)
        return 2

    try:
        config = Settings.from_env()
        if args.model:
            config = replace(config, default_model=args.model)
        if args.no_laya:
            config = replace(config, laya_enabled=False)
        task = build_orchestrator(config).run(request)
    except (ConfigurationError, OSError, ProviderError, ValueError) as exc:
        print(f"Aegis could not start: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        selected = ", ".join(task.selected_models) or "none"
        print(f"task={task.task_id} status={task.status} models={selected}", file=sys.stderr)
        for error in task.errors:
            print(f"error={error}", file=sys.stderr)

    if task.status is not TaskStatus.COMPLETED:
        print(task.errors[-1] if task.errors else f"Task ended with status: {task.status}", file=sys.stderr)
        return 1
    print(task.final_output or "")
    return 0
