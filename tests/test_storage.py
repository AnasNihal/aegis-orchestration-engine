from aegis_engine.storage import SQLiteTaskStateStore
from aegis_engine.tasks import TaskResult, TaskStatus


def test_sqlite_store_round_trips_task_state(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    state = store.create("Persist this task", task_id="task-1")
    completed = state.transition(TaskStatus.PLANNING).with_updates(
        plan=("step-one",),
        current_step="step-one",
        results=(TaskResult(step="step-one", success=True, output="done"),),
    )
    store.save(completed)

    loaded = store.get("task-1")

    assert loaded == completed
    assert store.list_recent(limit=1) == (completed,)


def test_sqlite_store_supports_delete(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    store.create("Delete this task", task_id="task-2")

    store.delete("task-2")

    assert store.list_recent() == ()
