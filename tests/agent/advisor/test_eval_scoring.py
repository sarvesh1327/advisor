from agent.advisor.eval_fixtures import EvalExpectation, EvalFixture, HumanReviewRubric
from agent.advisor.eval_scoring import score_advice_against_fixture
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, FocusTarget, RepoSummary


def _fixture() -> EvalFixture:
    return EvalFixture(
        fixture_id="coding-main-entry",
        domain="coding",
        description="fix main entrypoint bug",
        input_packet=AdvisorInputPacket(
            run_id="run-1",
            task_text="fix main entrypoint bug",
            task_type="bugfix",
            repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
            repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
            candidate_files=[CandidateFile(path="main.py", reason="entrypoint", score=0.9)],
            recent_failures=[],
            constraints=[],
            tool_limits={},
            acceptance_criteria=["tests pass"],
            token_budget=600,
        ),
        expected_advice=EvalExpectation(
            focus_targets=["main.py"],
            anti_targets=["docs/brief.md"],
            required_plan_steps=["inspect main.py"],
            forbidden_plan_steps=["broad refactor"],
            expected_failure_modes=["editing an unrelated file"],
        ),
        human_review_rubric=HumanReviewRubric(scale=[0, 1, 2, 3], criteria=["helpfulness"]),
    )


def test_score_advice_against_fixture_reports_generic_metrics():
    advice = AdviceBlock(
        task_type="bugfix",
        focus_targets=[FocusTarget(kind="file", locator="main.py", rationale="entrypoint", priority=1)],
        recommended_plan=["inspect main.py", "run targeted test"],
        likely_failure_modes=["editing an unrelated file"],
        confidence=0.8,
    )

    score = score_advice_against_fixture(advice, _fixture())

    assert score["focus_target_precision"] == 1.0
    assert score["focus_target_recall"] == 1.0
    assert score["anti_target_violation_rate"] == 0.0
    assert score["required_plan_coverage"] == 1.0
    assert score["forbidden_plan_violation_rate"] == 0.0
    assert score["failure_mode_coverage"] == 1.0
    assert score["overall_score"] == 1.0
