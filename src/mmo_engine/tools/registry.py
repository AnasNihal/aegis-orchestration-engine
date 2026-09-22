"""Central registry for explicitly exposed tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Callable, Mapping


class PermissionClass(StrEnum):
    PURE = "pure"
    READ_ONLY = "read_only"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ToolRegistryError(ValueError):
    """Raised when a tool definition is invalid or duplicated."""


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler
    permission: PermissionClass = PermissionClass.PURE
    timeout_seconds: float = 5.0

    def provider_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.input_schema),
            },
        }


_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ToolRegistry:
    """Store tools separately from their execution policy."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition, *, replace_existing: bool = False) -> None:
        if not _TOOL_NAME.fullmatch(tool.name):
            raise ToolRegistryError("tool names must be 2-64 lowercase characters, digits, or underscores")
        if not tool.description.strip():
            raise ToolRegistryError(f"tool description is required: {tool.name}")
        if not callable(tool.handler):
            raise ToolRegistryError(f"tool handler is not callable: {tool.name}")
        if tool.timeout_seconds <= 0:
            raise ToolRegistryError(f"tool timeout must be positive: {tool.name}")
        if tool.input_schema.get("type") != "object":
            raise ToolRegistryError(f"tool input schema must be an object: {tool.name}")
        if tool.name in self._tools and not replace_existing:
            raise ToolRegistryError(f"tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool: {name}") from exc

    def all(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def provider_schemas(self, names: set[str] | frozenset[str] | None = None) -> list[dict[str, Any]]:
        tools = self.all() if names is None else tuple(self.get(name) for name in sorted(names))
        return [tool.provider_schema() for tool in tools]


def validate_arguments(schema: Mapping[str, Any], arguments: Any) -> None:
    """Validate the small JSON-schema subset used by local tools."""

    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    _validate_value(schema, arguments, path="$", root=True)


def _validate_value(schema: Mapping[str, Any], value: Any, *, path: str, root: bool = False) -> None:
    expected = schema.get("type")
    type_matches = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected in type_matches and not type_matches[expected](value):
        raise ValueError(f"{path} must be {expected}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has an unsupported value")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path} is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} is above the maximum")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ValueError(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            unexpected = set(value) - set(properties)
            if unexpected:
                raise ValueError(f"{path} contains unsupported fields")
        for name, item in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, dict):
                _validate_value(child_schema, item, path=f"{path}.{name}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_value(schema["items"], item, path=f"{path}[{index}]")

