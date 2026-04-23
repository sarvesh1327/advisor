from agent.advisor.operators.operator_runtime import (
    OperatorJobQueue,
    PromoteCheckpointJobPayload,
    run_continuous_training_cycle,
    run_operator_job,
)
from agent.advisor.training.hardening import build_phase6_hardening_report


def _rollout_result(
    *,
    rollout_id: str,
    profile_id: str = "coding-default",
    packet_run_id: str = "run-1",
    plan: list[str] | None = None,
    total_reward: float = 0.8,
):
    return {
        "rollout_id": rollout_id,
        "advisor_profile_id": profile_id,
        "packet": {
            "run_id": packet_run_id,
            "task_text": "repair main flow",
            "task_type": "bugfix",
            "repo": {"path": "/tmp/repo", "branch": "main", "dirty": False, "session_id": f"sess-{packet_run_id}"},
            "repo_summary": {"modules": ["app"], "hotspots": ["main.py"], "file_tree_slice": ["main.py"]},
            "candidate_files": [{"path": "main.py", "reason": "token overlap", "score": 0.9}],
            "recent_failures": [],
            "constraints": ["tests must pass"],
            "tool_limits": {"terminal": True},
            "acceptance_criteria": ["repair succeeds"],
            "token_budget": 900,
        },
        "primary_advice": {
            "task_type": "bugfix",
            "focus_targets": [{"locator": "main.py", "why": "entrypoint", "priority": 1}],
            "recommended_plan": plan or ["inspect main.py"],
            "confidence": 0.9,
        },
        "executor_result": {"status": "success", "summary": "patched main.py", "output": "patched main.py"},
        "verifier_results": [],
        "outcome": {"run_id": packet_run_id, "status": "success", "retries": 0, "tests_run": ["pytest -q"]},
        "reward_label": {"total_reward": total_reward, "quality_score": total_reward},
        "diagnostics": {"multi_turn": False},
        "multi_turn_transcript": [],
    }



def test_build_phase6_hardening_report_flags_duplicate_leakage_reward_mismatch_and_flat_rewards():
    rollout_group = {
        "group_id": "group-hardening",
        "advisor_profile_id": "coding-default",
        "results": [
            _rollout_result(rollout_id="rollout-1", packet_run_id="run-1", total_reward=0.8),
            _rollout_result(rollout_id="rollout-2", packet_run_id="run-2", total_reward=0.8),
        ],
        "reward_values": [0.8, 0.6],
        "summary": {},
    }

    report = build_phase6_hardening_report(rollout_group=rollout_group, advisor_profile_id="coding-default")
    issue_codes = {issue["code"] for issue in report["issues"]}

    assert report["blocking"] is True
    assert report["summary"]["rollout_count"] == 2
    assert report["summary"]["duplicate_signature_count"] == 1
    assert report["summary"]["distinct_reward_count"] == 1
    assert {"duplicate_training_signature", "reward_value_mismatch", "flat_reward_distribution"} <= issue_codes



def test_run_continuous_training_cycle_rejects_blocking_hardening_issues_before_training():
    queue = OperatorJobQueue("/tmp/phase6-hardening-cycle-jobs.json")
    bad_rollout_group = {
        "group_id": "group-bad-cycle",
        "advisor_profile_id": "coding-default",
        "results": [
            _rollout_result(rollout_id="rollout-1", packet_run_id="run-1", total_reward=0.7),
            _rollout_result(rollout_id="rollout-2", packet_run_id="run-2", total_reward=0.7),
        ],
        "reward_values": [0.7, 0.7],
        "summary": {},
    }

    try:
        run_continuous_training_cycle(
            queue,
            experiment_id="exp-phase6",
            advisor_profile_id="coding-default",
            rollout_group=bad_rollout_group,
            benchmark_manifests=[],
            train_profile_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("train should not run")),
        )
    except ValueError as exc:
        assert "blocking rollout hardening issues" in str(exc)
    else:
        raise AssertionError("expected blocking rollout hardening issues to stop the cycle")

    assert queue.list_jobs() == []



def test_run_operator_job_blocks_promotion_when_eval_evidence_is_regressive_or_malformed(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    missing_deltas_job = queue.enqueue_job(
        job_type="promote-checkpoint",
        payload=PromoteCheckpointJobPayload(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="ckpt-missing-deltas",
            evaluation={
                "advisor_profile_id": "coding-default",
                "candidate_checkpoint_id": "ckpt-missing-deltas",
                "promote": True,
                "decision_reason": "missing deltas should block",
            },
        ).model_dump(),
        resume_token="promote-missing-deltas",
    )
    blocked_missing_deltas = run_operator_job(
        queue,
        missing_deltas_job.job_id,
        promote_checkpoint_fn=lambda payload: {"promoted": True},
    )

    promote_job = queue.enqueue_job(
        job_type="promote-checkpoint",
        payload=PromoteCheckpointJobPayload(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="ckpt-bad",
            evaluation={
                "advisor_profile_id": "coding-default",
                "candidate_checkpoint_id": "ckpt-bad",
                "promote": True,
                "rollback": True,
                "deltas": {"overall_score": -0.1, "focus_target_recall": -0.2},
                "decision_reason": "regression detected",
            },
        ).model_dump(),
        resume_token="promote-bad",
    )

    result = run_operator_job(queue, promote_job.job_id, promote_checkpoint_fn=lambda payload: {"promoted": True})

    assert blocked_missing_deltas.status == "completed"
    assert blocked_missing_deltas.result["status"] == "blocked"
    assert blocked_missing_deltas.result["promoted"] is False
    assert "missing or non-finite" in blocked_missing_deltas.result["reason"]
    assert result.status == "completed"
    assert result.result["status"] == "blocked"
    assert result.result["promoted"] is False
    assert "regression detected" in result.result["reason"]
