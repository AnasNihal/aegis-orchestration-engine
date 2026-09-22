"""Immutable task state and a small in-memory store for the first runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from uuid import uuid4


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStateError(ValueError):
    """Raised when a task lifecycle transition is invalid."""


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.PLANNING: frozenset(
        {
            TaskStatus.EXECUTING,
            TaskStatus.WAITING_FOR_INPUT,
            TaskStatus.WAITING_FOR_CONFIRMATION,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.EXECUTING: frozenset(
        {
            TaskStatus.PLANNING,
            TaskStatus.WAITING_FOR_INPUT,
            TaskStatus.WAITING_FOR_CONFIRMATION,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.WAITING_FOR_INPUT: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.WAITING_FOR_CONFIRMATION: frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TaskResult:
    step: str
    success: bool
    output: str = ""
    model: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TaskState:
    task_id: str
    user_request: str
    status: TaskStatus = TaskStatus.PENDING
    phase: str = "created"
    plan: tuple[str, ...] = ()
    current_step: str | None = None
    completed_steps: tuple[str, ...] = ()
    results: tuple[TaskResult, ...] = ()
    errors: tuple[str, ...] = ()
    retry_count: int = 0
    selected_models: tuple[str, ...] = ()
    final_output: str | None = None
    cancellation_requested: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def transition(self, status: TaskStatus, *, phase: str | None = None) -> "TaskState":
        if status != self.status and status not in _ALLOWED_TRANSITIONS[self.status]:
            raise TaskStateError(f"invalid task transition: {self.status} -> {status}")
        return replace(self, status=status, phase=phase or self.phase, updated_at=_now())

    def with_updates(self, **changes: object) -> "TaskState":
        return replace(self, **changes, updated_at=_now())


class InMemoryTaskStateStore:
    """Thread-safe volatile storage used until SQLite persistence is added."""

    def __init__(self) -> None:
        self._states: dict[str, TaskState] = {}
        self._lock = RLock()

    def create(self, user_request: str, *, task_id: str | None = None) -> TaskState:
        if not user_request.strip():
            raise TaskStateError("user_request must not be empty")
        state = TaskState(task_id=task_id or str(uuid4()), user_request=user_request)
        with self._lock:
            if state.task_id in self._states:
                raise TaskStateError(f"task already exists: {state.task_id}")
            self._states[state.task_id] = state
        return state

    def save(self, state: TaskState) -> TaskState:
        with self._lock:
            if state.task_id not in self._states:
                raise TaskStateError(f"task does not exist: {state.task_id}")
            self._states[state.task_id] = state
        return state

    def get(self, task_id: str) -> TaskState:
        with self._lock:
            try:
                return self._states[task_id]
            except KeyError as exc:
                raise TaskStateError(f"task does not exist: {task_id}") from exc

