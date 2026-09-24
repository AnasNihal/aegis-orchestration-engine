"""Evidence-based completion checks for task state."""

from __future__ import annotations

from dataclasses import dataclass

from aegis_engine.tasks.state import TaskState, TaskStatus


@dataclass(frozen=True)
class VerificationReport:
    verified: bool
    status: str
    evidence: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


def verify_task(state: TaskState) -> VerificationReport:
    """Verify a completed state using application evidence only.

    A model's assertion that it succeeded is never used as evidence. Tool
    results and the task lifecycle state are the source of truth.
    """

    issues: list[str] = []
    evidence: list[str] = []
    if state.status is not TaskStatus.COMPLETED:
        issues.append(f"task status is {state.status}, not completed")
    if not state.final_output or not state.final_output.strip():
        issues.append("task has no final output")
    else:
        evidence.append("final output is present")
    failed_results = [result.step for result in state.results if not result.success]
    if failed_results:
        issues.append(f"failed execution results: {', '.join(failed_results)}")
    elif state.results:
        evidence.append("recorded execution results are successful")
    if state.cancellation_requested:
        issues.append("task requested cancellation")
    return VerificationReport(
        verified=not issues,
        status="verified" if not issues else "unverified",
        evidence=tuple(evidence),
        issues=tuple(issues),
    )
