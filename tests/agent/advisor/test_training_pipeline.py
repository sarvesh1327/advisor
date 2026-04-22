from agent.advisor.reward_model import compute_reward_label
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, CandidateFile, RepoSummary
from agent.advisor.trace_store import AdvisorTraceStore
from agent.advisor.training_pipeline import (
    AblationSpec,
    ExperimentConfig,
    RollbackPolicy,
    build_dataset_manifest,
    evaluate_checkpoint,
    generate_ablation_plans,
    should_rollback,
)


def _packet(run_id: str, repo_path: str = "/tmp/repo", task_text: str = "fix prompt builder"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text=task_text,
        task_type="bugfix",
        repo={"path": repo_path, "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests pass"],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=900,
    )


def _record_run(store: AdvisorTraceStore, run_id: str, *, repo_path: str = "/tmp/repo", task_text: str = "fix prompt builder", status: str = "success"):
    packet = _packet(run_id, repo_path=repo_path, task_text=task_text)
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(
        run_id=run_id,
        status=status,
        files_touched=["main.py"],
        retries=0 if status == "success" else 3,
        tests_run=["pytest -q"] if status == "success" else [],
        review_verdict="pass" if status == "success" else "regressed",
    )
    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash=f"hash-{run_id}")
    store.record_outcome(outcome)
    store.record_reward_label(compute_reward_label(packet, advice, outcome, human_rating=5.0 if status == "success" else 1.0))


def test_build_dataset_manifest_groups_examples_and_supports_both_training_modes(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    _record_run(store, "run-a", repo_path="/tmp/repo-a", task_text="fix login prompt", status="success")
    _record_run(store, "run-b", repo_path="/tmp/repo-b", task_text="fix layout prompt", status="failure")

    config = ExperimentConfig(
        experiment_id="exp-phase9",
        student_model="advisor-small",
        target_executor="gpt-4.1",
        domain_mix={"coding": 1.0},
        training_mode="supervised",
        preference_training_mode="dpo",
    )

    manifest = build_dataset_manifest(store, config, min_quality_score=0.5)

    assert manifest["experiment_id"] == "exp-phase9"
    assert manifest["training_mode"] == "supervised"
    assert manifest["preference_training_mode"] == "dpo"
    assert manifest["counts"]["total_examples"] == 2
    assert manifest["counts"]["positive_examples"] == 1
    assert manifest["counts"]["negative_examples"] == 1
    assert set(manifest["splits"]) <= {"train", "val", "test"}


def test_evaluate_checkpoint_reports_transfer_and_regression_signals():
    config = ExperimentConfig(
        experiment_id="exp-phase9",
        student_model="advisor-small",
        target_executor="claude-sonnet",
        transfer_executor="gpt-4.1",
        domain_mix={"coding": 0.7, "research": 0.3},
        rollback=RollbackPolicy(min_success_delta=-0.02, min_score_delta=-0.03),
    )

    report = evaluate_checkpoint(
        config,
        checkpoint_name="ckpt-3",
        baseline_metrics={"success_rate": 0.72, "mean_score": 0.68},
        candidate_metrics={"success_rate": 0.69, "mean_score": 0.6},
        transfer_metrics={"success_rate": 0.71, "mean_score": 0.66},
    )

    assert report["checkpoint_name"] == "ckpt-3"
    assert report["target_executor"] == "claude-sonnet"
    assert report["transfer_executor"] == "gpt-4.1"
    assert report["deltas"]["success_rate"] == -0.03
    assert report["rollback"] is True


def test_generate_ablation_plans_covers_packet_advice_and_reward_components():
    config = ExperimentConfig(
        experiment_id="exp-ablation",
        student_model="advisor-small",
        target_executor="gpt-4.1",
        domain_mix={"coding": 1.0},
        ablations=[
            AblationSpec(kind="packet_field", target="history"),
            AblationSpec(kind="advice_field", target="focus_targets"),
            AblationSpec(kind="reward_component", target="human_usefulness"),
        ],
    )

    plans = generate_ablation_plans(config)

    assert [plan["kind"] for plan in plans] == ["packet_field", "advice_field", "reward_component"]
    assert plans[0]["variant_id"] == "packet_field:history"
    assert plans[2]["variant_id"] == "reward_component:human_usefulness"


def test_should_rollback_uses_policy_thresholds():
    policy = RollbackPolicy(min_success_delta=-0.01, min_score_delta=-0.02)

    assert should_rollback(policy, {"success_rate": -0.02, "mean_score": 0.0}) is True
    assert should_rollback(policy, {"success_rate": 0.0, "mean_score": -0.03}) is True
    assert should_rollback(policy, {"success_rate": 0.02, "mean_score": -0.01}) is False
