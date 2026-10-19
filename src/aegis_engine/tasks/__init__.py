"""Task lifecycle state and storage contracts."""

from .state import (
    InMemoryTaskStateStore,
    TaskResult,
    TaskState,
    TaskStateError,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
)


def __getattr__(name: str):
    # Preserve the public alias without importing storage while its own module
    # is still loading task state.
    if name == "SQLiteTaskStateStore":
        from aegis_engine.storage.sqlite import SQLiteTaskStateStore

        return SQLiteTaskStateStore
    raise AttributeError(name)

__all__ = [
    "InMemoryTaskStateStore",
    "TaskResult",
    "TaskState",
    "TaskStateError",
    "TaskStatus",
    "TaskStep",
    "TaskStepStatus",
    "SQLiteTaskStateStore",
]
