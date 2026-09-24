from aegis_engine.tasks import TaskResult, TaskStatus, TaskState
from aegis_engine.verification import ErrorCategory, classify_error, verify_task
from aegis_engine.models import ProviderError


def completed_state() -> TaskState:
    state = TaskState(task_id="task", user_request="answer")
    state = state.transition(TaskStatus.PLANNING).transition(TaskStatus.EXECUTING)
    return state.with_updates(
        results=(TaskResult(step="generate_response", success=True, output="answer"),),
        final_output="answer",
    ).transition(TaskStatus.COMPLETED)


def test_verifier_accepts_evidence_backed_completion() -> None:
    report = verify_task(completed_state())

    assert report.verified is True
    assert report.status == "verified"


def test_verifier_rejects_failed_execution_result() -> None:
    state = completed_state().with_updates(
        results=(TaskResult(step="tool:read", success=False, error="permission denied"),)
    )

    report = verify_task(state)

    assert report.verified is False
    assert "failed execution results" in report.issues[0]


def test_error_classifier_preserves_retry_signal() -> None:
    assert classify_error(ProviderError("temporary", provider="test", retryable=True)) is ErrorCategory.TRANSIENT
    assert classify_error("Tool execution timed out") is ErrorCategory.TIMEOUT
