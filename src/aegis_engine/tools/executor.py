"""Permission-aware, bounded tool execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Mapping

from aegis_engine.models.base import ToolCall
from aegis_engine.tools.registry import (
    PermissionClass,
    ToolDefinition,
    ToolRegistry,
    validate_arguments,
)


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    permission: PermissionClass | None = None
    timed_out: bool = False
    duration_ms: int = 0
    requires_confirmation: bool = False


class ToolExecutor:
    """Execute only registered and explicitly allowed tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tools: set[str] | frozenset[str] | None = None,
        confirmation_handler: Callable[[ToolDefinition, Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_tools = None if allowed_tools is None else frozenset(allowed_tools)
        self.confirmation_handler = confirmation_handler

    def execute_call(self, call: ToolCall) -> ToolResult:
        return self.execute(call.name, call.arguments)

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        started = monotonic()
        try:
            tool = self.registry.get(name)
        except Exception as exc:
            return self._failure(name, f"Tool unavailable: {exc}", started=started)
        args = dict(arguments or {})
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return self._failure(name, "Tool is not permitted for this task", tool=tool, started=started)
        try:
            validate_arguments(tool.input_schema, args)
        except ValueError as exc:
            return self._failure(name, f"Invalid tool arguments: {exc}", tool=tool, started=started)
        if tool.permission is PermissionClass.CONFIRMATION_REQUIRED:
            confirmed = self.confirmation_handler is not None and self.confirmation_handler(tool, args)
            if not confirmed:
                return self._failure(
                    name,
                    "User confirmation is required before this tool can run",
                    tool=tool,
                    started=started,
                    requires_confirmation=True,
                )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
        future = executor.submit(tool.handler, **args)
        try:
            output = future.result(timeout=tool.timeout_seconds)
            return ToolResult(
                tool_name=name,
                success=True,
                output=output,
                permission=tool.permission,
                duration_ms=int((monotonic() - started) * 1000),
            )
        except FutureTimeoutError:
            future.cancel()
            return self._failure(
                name,
                "Tool execution timed out",
                tool=tool,
                started=started,
                timed_out=True,
            )
        except Exception:
            return self._failure(
                name,
                "Tool execution failed",
                tool=tool,
                started=started,
            )
        finally:
            # A running Python thread cannot be forcibly killed. The registry's
            # initial tools are bounded pure/read-only functions; later tools
            # must use subprocess isolation if hard cancellation is required.
            executor.shutdown(wait=False, cancel_futures=True)

    def _failure(
        self,
        name: str,
        error: str,
        *,
        tool: ToolDefinition | None = None,
        started: float,
        timed_out: bool = False,
        requires_confirmation: bool = False,
    ) -> ToolResult:
        return ToolResult(
            tool_name=name,
            success=False,
            error=error,
            permission=tool.permission if tool else None,
            timed_out=timed_out,
            duration_ms=int((monotonic() - started) * 1000),
            requires_confirmation=requires_confirmation,
        )

