import time
from pathlib import Path

import pytest

from aegis_engine.models import ToolCall
from aegis_engine.tools import (
    PermissionClass,
    ToolDefinition,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryError,
    build_builtin_registry,
)


def test_builtin_tools_are_registered_with_provider_schemas() -> None:
    registry = build_builtin_registry()

    assert {tool["function"]["name"] for tool in registry.provider_schemas()} == {
        "calculator",
        "current_time",
        "read_text_file",
        "word_count",
    }


def test_executor_validates_arguments_and_executes_safe_tool() -> None:
    executor = ToolExecutor(build_builtin_registry())

    result = executor.execute_call(ToolCall(name="calculator", arguments={"expression": "2 + 3 * 4"}))
    invalid = executor.execute("calculator", {"expression": "__import__('os')"})

    assert result.success is True
    assert result.output == {"value": 14}
    assert invalid.success is False
    assert invalid.error and "failed" in invalid.error.lower()


def test_executor_enforces_allowlist_and_read_root(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    allowed = approved / "note.txt"
    allowed.write_text("private but approved", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("not approved", encoding="utf-8")
    registry = build_builtin_registry((str(approved),))

    executor = ToolExecutor(registry, allowed_tools={"read_text_file"})
    allowed_result = executor.execute("read_text_file", {"path": str(allowed)})
    outside_result = executor.execute("read_text_file", {"path": str(outside)})
    denied_result = executor.execute("calculator", {"expression": "1 + 1"})

    assert allowed_result.success is True
    assert allowed_result.output["content"] == "private but approved"
    assert outside_result.success is False
    assert denied_result.success is False
    assert "not permitted" in (denied_result.error or "")


def test_confirmation_required_tool_waits_without_confirmation() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="confirm_action",
            description="Test confirmation boundary.",
            input_schema={"type": "object", "additionalProperties": False},
            handler=lambda: "done",
            permission=PermissionClass.CONFIRMATION_REQUIRED,
        )
    )

    result = ToolExecutor(registry).execute("confirm_action")

    assert result.success is False
    assert result.requires_confirmation is True


def test_executor_enforces_timeout() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="slow_action",
            description="Test timeout boundary.",
            input_schema={"type": "object", "additionalProperties": False},
            handler=lambda: time.sleep(0.1),
            timeout_seconds=0.01,
        )
    )

    result = ToolExecutor(registry).execute("slow_action")

    assert result.success is False
    assert result.timed_out is True


def test_registry_rejects_duplicates_and_bad_names() -> None:
    registry = ToolRegistry()
    tool = ToolDefinition(
        name="safe_tool",
        description="A safe test tool.",
        input_schema={"type": "object"},
        handler=lambda: None,
    )
    registry.register(tool)

    with pytest.raises(ToolRegistryError):
        registry.register(tool)
    with pytest.raises(ToolRegistryError):
        registry.register(
            ToolDefinition(
                name="bad-name",
                description="Invalid name.",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
        )

