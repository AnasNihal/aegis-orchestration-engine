from aegis_engine.storage import SQLiteTaskStateStore
import pytest

from aegis_engine.tasks import TaskResult, TaskStateError, TaskStatus, TaskStep, TaskStepStatus


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


def test_pending_cancellation_prevents_claim(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    store.create("queued", task_id="queued")

    cancelled = store.request_cancel("queued")

    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.cancellation_requested
    with pytest.raises(TaskStateError, match="not claimable"):
        store.claim("queued")


def test_active_cancellation_survives_stale_state_save(tmp_path) -> None:
    store = SQLiteTaskStateStore(tmp_path / "tasks.sqlite3")
    active = store.claim(store.create("running", task_id="running").task_id)
    store.request_cancel("running")

    saved = store.save(active.with_updates(phase="inference"))

    assert saved.cancellation_requested
    assert store.get("running").cancellation_requested


def test_restart_recovery_closes_interrupted_tasks(tmp_path) -> None:
    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStateStore(path)
    store.create("queued", task_id="queued")
    active = store.claim(store.create("running", task_id="running").task_id)
    store.save(active.with_updates(
        status=TaskStatus.EXECUTING,
        current_step="generate_response",
        steps=(TaskStep("generate_response", TaskStepStatus.ACTIVE),),
    ))
    store.create("cancel me", task_id="cancel-me")
    store.request_cancel("cancel-me")

    restarted = SQLiteTaskStateStore(path)
    assert restarted.recover_interrupted() == 2
    assert restarted.recover_interrupted() == 0
    assert restarted.get("queued").status is TaskStatus.FAILED
    running = restarted.get("running")
    assert running.status is TaskStatus.FAILED
    assert running.steps[0].status is TaskStepStatus.FAILED
    assert "server restart" in running.errors[-1]
    assert restarted.get("cancel-me").status is TaskStatus.CANCELLED

    with pytest.raises(TaskStateError, match="terminal"):
        restarted.save(active.with_updates(phase="late worker save"))
    assert restarted.get("running").status is TaskStatus.FAILED
