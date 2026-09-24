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
from aegis_engine.tasks import SQLiteTaskStateStore, TaskStatus


def build_orchestrator(config: Settings) -> Orchestrator:
    """Discover local Ollama models and construct the bounded runtime."""

    provider = OllamaProvider(config)
    registry = ModelRegistry()
    registry.refresh(provider)
    gateway = ModelGateway([provider])
    router = DeterministicModelRouter(registry, preferred_model=config.default_model)
    return Orchestrator(
        gateway,
        router,
        provider_settings=config,
        store=SQLiteTaskStateStore(config.task_db_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Aegis orchestration task.")
    parser.add_argument("request", nargs="?", help="The task to execute.")
    parser.add_argument("--model", help="Prefer a specific installed Ollama model.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the session open for multiple questions.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the local browser interface.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web server bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Web server port.")
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


def run_interactive(config: Settings, *, verbose: bool = False) -> int:
    """Run a persistent local chat session using one configured model."""

    try:
        orchestrator = build_orchestrator(config)
    except (ConfigurationError, OSError, ProviderError, ValueError) as exc:
        print(f"Aegis could not start: {exc}", file=sys.stderr)
        return 1

    print(f"Aegis interactive mode | model={config.default_model}")
    print("Type /exit or /quit to stop.")
    while True:
        try:
            request = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if request.lower() in {"/exit", "/quit", "exit", "quit"}:
            return 0
        if not request:
            continue

        task = orchestrator.run(request)
        if task.status is TaskStatus.COMPLETED:
            print(f"Aegis> {task.final_output or ''}")
        else:
            print(
                f"Aegis task failed: {task.errors[-1] if task.errors else task.status}",
                file=sys.stderr,
            )
        if verbose:
            selected = ", ".join(task.selected_models) or "none"
            print(f"[task={task.task_id} status={task.status} models={selected}]", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.interactive and not args.serve and not args.request:
        request = input("Aegis> ").strip()
    else:
        request = args.request or ""
    if not request and not args.interactive and not args.serve:
        print("A request is required.", file=sys.stderr)
        return 2

    try:
        config = Settings.from_env()
        if args.model:
            config = replace(config, default_model=args.model)
        if args.no_laya:
            config = replace(config, laya_enabled=False)
        if args.serve:
            from aegis_engine.web import serve

            serve(config, host=args.host, port=args.port)
            return 0
        if args.interactive:
            return run_interactive(config, verbose=args.verbose)
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
