import pytest

from aegis_engine.orchestration import DeterministicPlanner, PlanningError


def test_planner_creates_bounded_direct_plan() -> None:
    plan = DeterministicPlanner().create("Explain this")

    assert plan.steps == ("generate_response",)
    assert plan.reason == "direct response plan"


def test_planner_records_selected_tool_phase() -> None:
    plan = DeterministicPlanner().create("Calculate this", tool_names=("calculator",))

    assert plan.steps == ("select_tools", "generate_response", "execute_tools", "generate_response")


def test_planner_rejects_plan_over_limit() -> None:
    with pytest.raises(PlanningError):
        DeterministicPlanner(max_steps=2).create("Use a tool", tool_names=("calculator",))
