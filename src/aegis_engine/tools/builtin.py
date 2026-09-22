"""Safe built-in tools for the first execution layer."""

from __future__ import annotations

import ast
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Callable

from aegis_engine.tools.registry import PermissionClass, ToolDefinition, ToolRegistry


def calculator(expression: str) -> dict[str, int | float]:
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")
    value = _evaluate(tree.body)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not finite")
    return {"value": value}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent is too large")
        return {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }[type(node.op)]()
    raise ValueError("expression contains an unsupported operation")


def current_time() -> dict[str, str]:
    now = datetime.now().astimezone()
    return {"iso": now.isoformat(), "display": now.strftime("%A, %B %-d %Y at %-I:%M %p %Z")}


def word_count(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
    }


def _read_text_file(path: str, roots: tuple[Path, ...]) -> dict[str, str]:
    candidate = Path(path).expanduser().resolve()
    if not roots or not any(candidate == root or root in candidate.parents for root in roots):
        raise PermissionError("path is outside approved file roots")
    if not candidate.is_file():
        raise FileNotFoundError("approved file does not exist")
    if candidate.stat().st_size > 128 * 1024:
        raise ValueError("file is larger than the 128 KiB limit")
    return {"path": str(candidate), "content": candidate.read_text(encoding="utf-8", errors="replace")}


def build_builtin_registry(approved_file_roots: tuple[str, ...] = ()) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Evaluate a basic arithmetic expression without executing code.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 200}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=calculator,
        )
    )
    registry.register(
        ToolDefinition(
            name="current_time",
            description="Return the current local time from the host system.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=current_time,
        )
    )
    registry.register(
        ToolDefinition(
            name="word_count",
            description="Count characters, words, and lines in supplied text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 100_000}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=word_count,
        )
    )
    roots = tuple(Path(root).expanduser().resolve() for root in approved_file_roots)
    registry.register(
        ToolDefinition(
            name="read_text_file",
            description="Read a small UTF-8 text file inside an explicitly approved root.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 1024}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=lambda path: _read_text_file(path, roots),
            permission=PermissionClass.READ_ONLY,
        )
    )
    return registry

