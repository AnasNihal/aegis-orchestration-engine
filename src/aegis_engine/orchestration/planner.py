"""Bounded deterministic planning for orchestration tasks."""

from __future__ import annotations

from dataclasses import dataclass


class PlanningError(ValueError):
    """Raised when a task cannot receive a safe bounded plan."""


@dataclass(frozen=True)
class TaskPlan:
    steps: tuple[str, ...]
    reason: str


class DeterministicPlanner:
    """Describe bounded orchestration phases without LLM-generated commands."""

    def __init__(self, *, max_steps: int = 5) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps

    def create(self, user_request: str, *, tool_names: tuple[str, ...] = ()) -> TaskPlan:
        if not user_request.strip():
            raise PlanningError("user_request must not be empty")
        steps = ("generate_response",)
        reason = "direct response plan"
        if tool_names:
            steps = ("select_tools", "generate_response", "execute_tools", "generate_response")
            reason = "bounded response and selected-tool plan"
        if len(steps) > self.max_steps:
            raise PlanningError("task plan exceeds the configured step limit")
        return TaskPlan(steps=steps, reason=reason)
