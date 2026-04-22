from agent.advisor.reward_model import RewardWeights, compute_reward_label
from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    CandidateFile,
    FocusTarget,
    RepoSummary,
)
from agent.advisor.settings import AdvisorSettings


def _packet(run_id: str = "run-1"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="fix prompt builder",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests pass"],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=900,
    )


def test_compute_reward_label_scores_canonical_components():
    packet = _packet()
    advice = AdviceBlock(
        task_type="bugfix",
        focus_targets=[FocusTarget(kind="file", locator="main.py", rationale="touch target", priority=1)],
        recommended_plan=["inspect main.py"],
        confidence=0.8,
    )
    outcome = AdvisorOutcome(
        run_id="run-1",
        status="success",
        files_touched=["main.py"],
        retries=1,
        tests_run=["pytest -q"],
        review_verdict="pass",
    )

    label = compute_reward_label(packet, advice, outcome, human_rating=4.0)

    assert label.run_id == "run-1"
    assert label.reward_version == "phase8-v1"
    assert label.components.task_success == 1.0
    assert label.components.targeting_quality == 1.0
    assert label.components.constraint_compliance == 1.0
    assert label.components.human_usefulness == 0.8
    assert 0.0 <= label.components.efficiency <= 1.0
    assert 0.0 <= label.total_reward <= 1.0
    assert label.example_type == "positive"
    assert label.hard_case_bucket is None


def test_compute_reward_label_marks_negative_hard_cases_from_failures():
    packet = _packet("run-failure")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect docs.md"], confidence=0.4)
    outcome = AdvisorOutcome(
        run_id="run-failure",
        status="failure",
        files_touched=["main.py"],
        retries=4,
        tests_run=[],
        review_verdict="constraint drift",
        summary="edited unrelated files",
    )

    label = compute_reward_label(
        packet,
        advice,
        outcome,
        constraint_violations=["edited unrelated files"],
        weights=RewardWeights(human_usefulness=0.0),
    )

    assert label.components.task_success == 0.0
    assert label.components.targeting_quality == 0.0
    assert label.components.constraint_compliance == 0.0
    assert label.example_type == "negative"
    assert label.hard_case_bucket == "constraint_failure"
    assert label.quality_score == label.total_reward


def test_compute_reward_label_accepts_configured_weight_presets():
    packet = _packet("run-human-first")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(
        run_id="run-human-first",
        status="success",
        files_touched=["main.py"],
        retries=2,
        tests_run=["pytest -q"],
        review_verdict="pass",
    )

    balanced = compute_reward_label(
        packet,
        advice,
        outcome,
        human_rating=1.0,
        weights=AdvisorSettings(reward_preset="balanced").reward_weights(),
    )
    human_first = compute_reward_label(
        packet,
        advice,
        outcome,
        human_rating=1.0,
        weights=AdvisorSettings(reward_preset="human-first").reward_weights(),
    )

    assert human_first.total_reward < balanced.total_reward
