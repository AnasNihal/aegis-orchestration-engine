"""Bounded in-process runner for submitted orchestration tasks.

This runner is intentionally single-process. SQLite records the outcome, but
interrupted work is marked failed on restart rather than replayed implicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import BoundedSemaphore, RLock
from uuid import uuid4

from aegis_engine.models.base import ChatMessage
from aegis_engine.orchestration.engine import Orchestrator
from aegis_engine.tasks.state import TaskResult, TaskState, TaskStatus, TaskStepStatus


class TaskCapacityError(RuntimeError):
    """Raised when all running and queued task slots are occupied."""


class TaskRunner:
    """Submit bounded work and persist every terminal outcome."""

    def __init__(self, orchestrator: Orchestrator, *, max_workers: int = 1, max_queued: int = 4) -> None:
        if max_workers < 1 or max_queued < 0:
            raise ValueError("max_workers must be positive and max_queued must not be negative")
        self.orchestrator = orchestrator
        self.store = orchestrator.store
        recover = getattr(self.store, "recover_interrupted", None)
        if callable(recover):
            recover()
        self._slots = BoundedSemaphore(max_workers + max_queued)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aegis-task")
        self._lock = RLock()
        self._futures: dict[str, Future[None]] = {}
        self._closed = False

    def submit(
        self,
        request: str,
        *,
        model_id: str,
        conversation: Sequence[ChatMessage] = (),
        tool_names: tuple[str, ...] = (),
    ) -> TaskState:
        with self._lock:
            if self._closed:
                raise TaskCapacityError("task runner is closed")
            if not self._slots.acquire(blocking=False):
                raise TaskCapacityError("task capacity reached; try again later")
            task_id = str(uuid4())
            try:
                state = self.store.create(request, task_id=task_id)
                future = self._pool.submit(
                    self._run, request, task_id, model_id, tuple(conversation), tool_names
                )
                self._futures[task_id] = future
                future.add_done_callback(lambda completed: self._finished(task_id, completed))
                return state
            except Exception:
                self._slots.release()
                if "state" in locals():
                    self.store.request_cancel(task_id)
                raise

    def cancel(self, task_id: str) -> TaskState:
        with self._lock:
            state = self.store.request_cancel(task_id)
            future = self._futures.get(task_id)
            if state.status is TaskStatus.CANCELLED and future is not None:
                future.cancel()
            return state

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # Wait outside the lock: completion callbacks need it to release slots.
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _finished(self, task_id: str, future: Future[None]) -> None:
        with self._lock:
            try:
                self._futures.pop(task_id, None)
                if future.cancelled():
                    self.store.request_cancel(task_id)
            finally:
                self._slots.release()

    def _run(
        self,
        request: str,
        task_id: str,
        model_id: str,
        conversation: tuple[ChatMessage, ...],
        tool_names: tuple[str, ...],
    ) -> None:
        try:
            self.orchestrator.run(
                request,
                task_id=task_id,
                is_cancelled=lambda: self.store.get(task_id).cancellation_requested,
                model_id=model_id,
                conversation=conversation,
                tool_names=tool_names,
            )
        except Exception:
            # Internal exception text may contain provider or filesystem data.
            # Record a stable, non-sensitive failure instead.
            error = "Background task failed unexpectedly"
            state = self.store.get(task_id)
            if state.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return
            if state.cancellation_requested:
                self.store.save(state.transition(TaskStatus.CANCELLED, phase="cancelled"))
                return
            if state.status is TaskStatus.PENDING:
                state = state.transition(TaskStatus.PLANNING)
            if state.status in {TaskStatus.WAITING_FOR_INPUT, TaskStatus.WAITING_FOR_CONFIRMATION}:
                state = state.transition(TaskStatus.PLANNING)
            steps = tuple(
                replace(step, status=TaskStepStatus.FAILED, error=error)
                if step.status is TaskStepStatus.ACTIVE else step
                for step in state.steps
            )
            state = state.with_updates(
                steps=steps,
                errors=(*state.errors, error),
                results=(*state.results, TaskResult(step=state.current_step or "task", success=False, error=error)),
            ).transition(TaskStatus.FAILED, phase="failed")
            self.store.save(state)
