from dataclasses import dataclass

from aegis_engine.cli import build_parser, main
from aegis_engine.tasks import TaskStatus


def test_cli_parser_accepts_request_and_runtime_options() -> None:
    args = build_parser().parse_args(["Summarize this", "--model", "qwen2.5:7b", "--no-laya"])

    assert args.request == "Summarize this"
    assert args.model == "qwen2.5:7b"
    assert args.no_laya is True


@dataclass
class FakeTask:
    status: TaskStatus = TaskStatus.COMPLETED
    task_id: str = "test-task"
    selected_models: tuple[str, ...] = ("fake/local",)
    errors: tuple[str, ...] = ()
    final_output: str = "done"


class FakeOrchestrator:
    def run(self, request: str) -> FakeTask:
        assert request == "hello"
        return FakeTask()


def test_cli_prints_completed_task(monkeypatch, capsys) -> None:
    monkeypatch.setattr("aegis_engine.cli.build_orchestrator", lambda config: FakeOrchestrator())

    assert main(["hello"]) == 0
    assert capsys.readouterr().out == "done\n"
