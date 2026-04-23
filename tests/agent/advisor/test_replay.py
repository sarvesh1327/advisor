from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, CandidateFile, FocusTarget, RepoSummary
from agent.advisor.evaluation.eval_fixtures import EvalExpectation, EvalFixture, HumanReviewRubric
from agent.advisor.evaluation.replay import evaluate_replay_run, list_replay_runs
from agent.advisor.storage.trace_store import AdvisorTraceStore


def _packet(run_id: str = "run-1"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="fix prompt builder",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=[],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=900,
    )


def _fixture() -> EvalFixture:
    return EvalFixture(
        fixture_id="coding-main-entry",
        domain="coding",
        description="fix prompt builder",
        input_packet=_packet("fixture-run"),
        expected_advice=EvalExpectation(
            focus_targets=["main.py"],
            anti_targets=["docs/brief.md"],
            required_plan_steps=["inspect main.py"],
            forbidden_plan_steps=["broad refactor"],
            expected_failure_modes=["editing an unrelated file"],
        ),
        human_review_rubric=HumanReviewRubric(scale=[0, 1, 2, 3], criteria=["helpfulness"]),
    )


def test_replay_lists_runs_and_scores_against_fixture(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet()
    advice = AdviceBlock(
        task_type="bugfix",
        focus_targets=[FocusTarget(kind="file", locator="main.py", rationale="entrypoint", priority=1)],
        recommended_plan=["inspect main.py"],
        likely_failure_modes=["editing an unrelated file"],
        confidence=0.8,
    )
    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="abc")
    store.record_outcome(AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=0, tests_run=["pytest -q"]))

    runs = list_replay_runs(store)
    scored = evaluate_replay_run(store, "run-1", _fixture())

    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["input"]["task"]["domain"] == "coding"
    assert scored["run_id"] == "run-1"
    assert scored["fixture_id"] == "coding-main-entry"
    assert scored["score"]["focus_target_recall"] == 1.0
