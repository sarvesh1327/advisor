import json

from agent.advisor.evaluation.measurement import build_phase5_measurement_report
from agent.advisor.operators.operator_runtime import OperatorJobQueue
from agent.advisor.training.training_runtime import (
    CheckpointLifecycleManager,
    TrainingCheckpointRecord,
)


def _write_checkpoint(
    manager: CheckpointLifecycleManager,
    *,
    profile_id: str,
    checkpoint_id: str,
    experiment_id: str,
    status: str,
    artifact_bytes: bytes,
    benchmark_summary: dict | None = None,
    rollback_reason: str | None = None,
):
    checkpoint_dir = manager.artifacts_root / "checkpoints" / profile_id / checkpoint_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / "adapters.safetensors"
    adapter_path.write_bytes(artifact_bytes)
    config_path = checkpoint_dir / "adapter_config.json"
    config_path.write_text(json.dumps({"checkpoint_id": checkpoint_id}, sort_keys=True), encoding="utf-8")
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "advisor_profile_id": profile_id,
                "artifact_paths": {
                    "adapter_model": str(adapter_path),
                    "adapter_config": str(config_path),
                },
                "training_metrics": {"optimizer_steps": 3},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
            path=str(checkpoint_dir),
            status=status,
            advisor_profile_id=profile_id,
            benchmark_summary=benchmark_summary or {},
            rollback_reason=rollback_reason,
        )
    )



def _enqueue_cycle_jobs(
    queue: OperatorJobQueue,
    *,
    experiment_id: str,
    profile_id: str,
    checkpoint_id: str,
    overall_delta: float,
    recall_delta: float,
    promote: bool,
    promotion_threshold: float = 0.05,
):
    queue.enqueue_job(
        job_type="train-profile",
        payload={
            "experiment_id": experiment_id,
            "advisor_profile_id": profile_id,
            "rollout_group": {
                "group_id": f"group-{checkpoint_id}",
                "advisor_profile_id": profile_id,
                "results": [],
                "reward_values": [],
                "summary": {},
            },
            "benchmark_manifests": [],
        },
        resume_token=f"continuous:{experiment_id}:{profile_id}:train",
    )
    train_job = queue.list_jobs()[-1]
    queue.update_job(
        train_job.job_id,
        status="completed",
        result={
            "checkpoint_id": checkpoint_id,
            "advisor_profile_id": profile_id,
        },
    )

    queue.enqueue_job(
        job_type="eval-profile",
        payload={
            "advisor_profile_id": profile_id,
            "candidate_checkpoint_id": checkpoint_id,
            "benchmark_manifests": [],
            "promotion_threshold": promotion_threshold,
        },
        resume_token=f"continuous:{experiment_id}:{profile_id}:eval:{checkpoint_id}",
    )
    eval_job = queue.list_jobs()[-1]
    queue.update_job(
        eval_job.job_id,
        status="completed",
        result={
            "advisor_profile_id": profile_id,
            "candidate_checkpoint_id": checkpoint_id,
            "promotion_threshold": promotion_threshold,
            "candidate_summary": {
                "overall_score": round(0.5 + overall_delta, 4),
                "focus_target_recall": round(0.5 + recall_delta, 4),
            },
            "baseline_summary": {
                "overall_score": 0.5,
                "focus_target_recall": 0.5,
            },
            "deltas": {
                "overall_score": overall_delta,
                "focus_target_recall": recall_delta,
            },
            "promote": promote,
            "rollback": overall_delta < 0.0 or recall_delta < 0.0,
            "decision_reason": f"decision for {checkpoint_id}",
        },
    )

    if promote:
        queue.enqueue_job(
            job_type="promote-checkpoint",
            payload={
                "advisor_profile_id": profile_id,
                "candidate_checkpoint_id": checkpoint_id,
                "evaluation": {
                    "advisor_profile_id": profile_id,
                    "candidate_checkpoint_id": checkpoint_id,
                    "promote": True,
                },
            },
            resume_token=f"continuous:{experiment_id}:{profile_id}:promote:{checkpoint_id}",
        )
        promote_job = queue.list_jobs()[-1]
        queue.update_job(
            promote_job.job_id,
            status="completed",
            result={
                "advisor_profile_id": profile_id,
                "checkpoint_id": checkpoint_id,
                "promoted": True,
                "status": "active",
            },
        )



def test_build_phase5_measurement_report_tracks_checkpoint_lineage_and_cycle_trends(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-1",
        experiment_id="exp-a",
        status="candidate",
        artifact_bytes=b"adapter-a",
        benchmark_summary={"overall_score": 0.61},
    )
    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-2",
        experiment_id="exp-b",
        status="active",
        artifact_bytes=b"adapter-b",
        benchmark_summary={"overall_score": 0.74},
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-a",
        profile_id="coding-default",
        checkpoint_id="ckpt-1",
        overall_delta=0.04,
        recall_delta=0.02,
        promote=False,
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-b",
        profile_id="coding-default",
        checkpoint_id="ckpt-2",
        overall_delta=0.18,
        recall_delta=0.12,
        promote=True,
    )

    report = build_phase5_measurement_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
    )

    coding = report["profiles"]["coding-default"]

    assert report["profile_count"] == 1
    assert coding["summary"]["checkpoint_count"] == 2
    assert coding["summary"]["cycle_count"] == 2
    assert coding["summary"]["promoted_cycle_count"] == 1
    assert coding["summary"]["active_checkpoint_id"] == "ckpt-2"
    assert [row["checkpoint_id"] for row in coding["checkpoint_history"]] == ["ckpt-1", "ckpt-2"]
    assert [row["experiment_id"] for row in coding["trend_history"]] == ["exp-a", "exp-b"]
    assert coding["trend_history"][1]["eval_delta"]["overall_score"] == 0.18
    assert coding["trend_history"][1]["promoted"] is True



def test_build_phase5_measurement_report_includes_artifact_fingerprints_and_change_flags(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-same-1",
        experiment_id="exp-same-1",
        status="candidate",
        artifact_bytes=b"same-adapter",
    )
    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-same-2",
        experiment_id="exp-same-2",
        status="candidate",
        artifact_bytes=b"same-adapter",
    )
    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-diff-3",
        experiment_id="exp-diff-3",
        status="active",
        artifact_bytes=b"different-adapter",
    )

    report = build_phase5_measurement_report(lifecycle_manager=manager, job_records=[])
    history = report["profiles"]["coding-default"]["checkpoint_history"]

    assert history[0]["artifact_fingerprint"]
    assert history[0]["artifact_changed_vs_previous"] is None
    assert history[1]["artifact_fingerprint"] == history[0]["artifact_fingerprint"]
    assert history[1]["artifact_changed_vs_previous"] is False
    assert history[2]["artifact_fingerprint"] != history[1]["artifact_fingerprint"]
    assert history[2]["artifact_changed_vs_previous"] is True



def test_build_phase5_measurement_report_keeps_profiles_isolated(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="coding-ckpt",
        experiment_id="exp-coding",
        status="active",
        artifact_bytes=b"coding-adapter",
    )
    _write_checkpoint(
        manager,
        profile_id="researcher",
        checkpoint_id="research-ckpt",
        experiment_id="exp-research",
        status="rolled_back",
        artifact_bytes=b"research-adapter",
        rollback_reason="regression detected",
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-coding",
        profile_id="coding-default",
        checkpoint_id="coding-ckpt",
        overall_delta=0.11,
        recall_delta=0.08,
        promote=True,
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-research",
        profile_id="researcher",
        checkpoint_id="research-ckpt",
        overall_delta=-0.07,
        recall_delta=-0.03,
        promote=False,
    )

    report = build_phase5_measurement_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
    )

    assert set(report["profiles"]) == {"coding-default", "researcher"}
    assert report["profiles"]["coding-default"]["summary"]["active_checkpoint_id"] == "coding-ckpt"
    assert report["profiles"]["researcher"]["summary"]["active_checkpoint_id"] is None
    assert report["profiles"]["researcher"]["checkpoint_history"][0]["rollback_reason"] == "regression detected"
    assert report["profiles"]["researcher"]["trend_history"][0]["promoted"] is False
    assert report["profiles"]["researcher"]["trend_history"][0]["eval_delta"]["overall_score"] == -0.07



def test_build_phase5_measurement_report_does_not_mark_promotion_without_completed_promote_job(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-pending-promote",
        experiment_id="exp-pending-promote",
        status="candidate",
        artifact_bytes=b"pending-promote",
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-pending-promote",
        profile_id="coding-default",
        checkpoint_id="ckpt-pending-promote",
        overall_delta=0.14,
        recall_delta=0.09,
        promote=False,
    )
    eval_job = queue.list_jobs()[-1]
    queue.update_job(
        eval_job.job_id,
        status="completed",
        result={
            "advisor_profile_id": "coding-default",
            "candidate_checkpoint_id": "ckpt-pending-promote",
            "promotion_threshold": 0.05,
            "candidate_summary": {"overall_score": 0.64, "focus_target_recall": 0.59},
            "baseline_summary": {"overall_score": 0.5, "focus_target_recall": 0.5},
            "deltas": {"overall_score": 0.14, "focus_target_recall": 0.09},
            "promote": True,
            "rollback": False,
            "decision_reason": "promote recommended but not completed",
        },
    )

    report = build_phase5_measurement_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
    )

    trend = report["profiles"]["coding-default"]["trend_history"][0]

    assert trend["promote_decision"] is True
    assert trend["promote_job_id"] is None
    assert trend["promoted"] is None
    assert report["profiles"]["coding-default"]["summary"]["promoted_cycle_count"] == 0



def test_build_phase5_measurement_report_prefers_latest_matching_completed_eval_job(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-latest-eval",
        experiment_id="exp-latest-eval",
        status="candidate",
        artifact_bytes=b"latest-eval",
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-latest-eval",
        profile_id="coding-default",
        checkpoint_id="ckpt-latest-eval",
        overall_delta=0.04,
        recall_delta=0.02,
        promote=False,
        promotion_threshold=0.05,
    )
    queue.enqueue_job(
        job_type="eval-profile",
        payload={
            "advisor_profile_id": "coding-default",
            "candidate_checkpoint_id": "ckpt-latest-eval",
            "benchmark_manifests": [],
            "promotion_threshold": 0.2,
        },
        resume_token="continuous:exp-latest-eval:coding-default:eval:retry",
    )
    latest_eval_job = queue.list_jobs()[-1]
    queue.update_job(
        latest_eval_job.job_id,
        status="completed",
        result={
            "advisor_profile_id": "coding-default",
            "candidate_checkpoint_id": "ckpt-latest-eval",
            "promotion_threshold": 0.2,
            "candidate_summary": {"overall_score": 0.77, "focus_target_recall": 0.7},
            "baseline_summary": {"overall_score": 0.5, "focus_target_recall": 0.5},
            "deltas": {"overall_score": 0.27, "focus_target_recall": 0.2},
            "promote": True,
            "rollback": False,
            "decision_reason": "latest eval should win",
        },
    )

    report = build_phase5_measurement_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
    )

    trend = report["profiles"]["coding-default"]["trend_history"][0]

    assert trend["eval_job_id"] == latest_eval_job.job_id
    assert trend["promotion_threshold"] == 0.2
    assert trend["eval_delta"]["overall_score"] == 0.27
    assert trend["promote_decision"] is True
