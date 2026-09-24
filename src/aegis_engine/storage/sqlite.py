"""SQLite-backed task state storage.

The store keeps task state separate from model conversation memory. It uses a
small versioned schema and JSON columns for the immutable task collections so
the execution model can evolve without introducing an ORM dependency.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from threading import RLock

from aegis_engine.tasks.state import TaskResult, TaskState, TaskStateError, TaskStatus


class SQLiteTaskStateStore:
    """Thread-safe local SQLite implementation of the task state store."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_request TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    current_step TEXT,
                    completed_steps_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    selected_models_json TEXT NOT NULL,
                    final_output TEXT,
                    cancellation_requested INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            if connection.execute("SELECT 1 FROM schema_version LIMIT 1").fetchone() is None:
                connection.execute("INSERT INTO schema_version(version) VALUES (1)")
            connection.commit()
            if self._connection is None:
                connection.close()

    def create(self, user_request: str, *, task_id: str | None = None) -> TaskState:
        from uuid import uuid4

        if not user_request.strip():
            raise TaskStateError("user_request must not be empty")
        state = TaskState(task_id=task_id or str(uuid4()), user_request=user_request)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    _state_values(state),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise TaskStateError(f"task already exists: {state.task_id}") from exc
            finally:
                if self._connection is None:
                    connection.close()
        return state

    def save(self, state: TaskState) -> TaskState:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    """UPDATE tasks SET user_request=?, status=?, phase=?, plan_json=?,
                    current_step=?, completed_steps_json=?, results_json=?, errors_json=?,
                    retry_count=?, selected_models_json=?, final_output=?,
                    cancellation_requested=?, created_at=?, updated_at=? WHERE task_id=?""",
                    (*_state_values(state)[1:], state.task_id),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise TaskStateError(f"task does not exist: {state.task_id}")
                connection.commit()
            finally:
                if self._connection is None:
                    connection.close()
        return state

    def get(self, task_id: str) -> TaskState:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            finally:
                if self._connection is None:
                    connection.close()
        if row is None:
            raise TaskStateError(f"task does not exist: {task_id}")
        return _row_to_state(row)

    def list_recent(self, *, limit: int = 50) -> tuple[TaskState, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            finally:
                if self._connection is None:
                    connection.close()
        return tuple(_row_to_state(row) for row in rows)

    def delete(self, task_id: str) -> None:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise TaskStateError(f"task does not exist: {task_id}")
                connection.commit()
            finally:
                if self._connection is None:
                    connection.close()


def _state_values(state: TaskState) -> tuple[object, ...]:
    return (
        state.task_id,
        state.user_request,
        state.status.value,
        state.phase,
        _json(state.plan),
        state.current_step,
        _json(state.completed_steps),
        _json([asdict(result) for result in state.results]),
        _json(state.errors),
        state.retry_count,
        _json(state.selected_models),
        state.final_output,
        int(state.cancellation_requested),
        state.created_at,
        state.updated_at,
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row_to_state(row: sqlite3.Row) -> TaskState:
    results = tuple(TaskResult(**item) for item in json.loads(row["results_json"]))
    return TaskState(
        task_id=row["task_id"],
        user_request=row["user_request"],
        status=TaskStatus(row["status"]),
        phase=row["phase"],
        plan=tuple(json.loads(row["plan_json"])),
        current_step=row["current_step"],
        completed_steps=tuple(json.loads(row["completed_steps_json"])),
        results=results,
        errors=tuple(json.loads(row["errors_json"])),
        retry_count=row["retry_count"],
        selected_models=tuple(json.loads(row["selected_models_json"])),
        final_output=row["final_output"],
        cancellation_requested=bool(row["cancellation_requested"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
