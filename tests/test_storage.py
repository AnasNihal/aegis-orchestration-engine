from aegis_engine.storage import SQLiteTaskStateStore
from aegis_engine.tasks import TaskResult, TaskStatus, TaskStep, TaskStepStatus


def test_sqlite_store_round_trips_task_state(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    state = store.create("Persist this task", task_id="task-1")
    completed = state.transition(TaskStatus.PLANNING).with_updates(
        plan=("step-one",),
        steps=(TaskStep("step-one", TaskStepStatus.COMPLETED),),
        current_step="step-one",
        results=(TaskResult(step="step-one", success=True, output="done"),),
    )
    store.save(completed)

    loaded = store.get("task-1")

    assert loaded == completed
    assert store.list_recent(limit=1) == (completed,)


def test_sqlite_store_migrates_old_schema_with_empty_checkpoints(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    import sqlite3

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (1);
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, user_request TEXT NOT NULL, status TEXT NOT NULL,
            phase TEXT NOT NULL, plan_json TEXT NOT NULL, current_step TEXT,
            completed_steps_json TEXT NOT NULL, results_json TEXT NOT NULL,
            errors_json TEXT NOT NULL, retry_count INTEGER NOT NULL,
            selected_models_json TEXT NOT NULL, final_output TEXT,
            cancellation_requested INTEGER NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    store = SQLiteTaskStateStore(path)
    assert store.create("Migrated task", task_id="legacy").steps == ()


def test_sqlite_store_supports_delete(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    store.create("Delete this task", task_id="task-2")

    store.delete("task-2")

    assert store.list_recent() == ()
