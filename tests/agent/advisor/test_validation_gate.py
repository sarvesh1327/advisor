import json

from agent.advisor.operators.operator_runtime import OperatorJobQueue
from agent.advisor.product.hardening import Phase8ValidationPolicy, build_phase8_validation_report
from agent.advisor.training.training_runtime import CheckpointLifecycleManager, TrainingCheckpointRecord


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
    rollback: bool,
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
        result={"checkpoint_id": checkpoint_id, "advisor_profile_id": profile_id},
    )

    queue.enqueue_job(
        job_type="eval-profile",
        payload={
            "advisor_profile_id": profile_id,
            "candidate_checkpoint_id": checkpoint_id,
            "benchmark_manifests": [],
            "promotion_threshold": 0.05,
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
            "promotion_threshold": 0.05,
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
            "rollback": rollback,
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
                    "deltas": {
                        "overall_score": overall_delta,
                        "focus_target_recall": recall_delta,
                    },
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


def test_build_phase8_validation_report_passes_with_promotion_rollback_and_positive_improvement(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-rollback",
        experiment_id="exp-a",
        status="rolled_back",
        artifact_bytes=b"adapter-a",
        benchmark_summary={"overall_score": 0.47},
        rollback_reason="regressed",
    )
    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        experiment_id="exp-b",
        status="active",
        artifact_bytes=b"adapter-b",
        benchmark_summary={"overall_score": 0.74},
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-a",
        profile_id="coding-default",
        checkpoint_id="ckpt-rollback",
        overall_delta=-0.08,
        recall_delta=-0.03,
        promote=False,
        rollback=True,
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-b",
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        overall_delta=0.19,
        recall_delta=0.12,
        promote=True,
        rollback=False,
    )

    report = build_phase8_validation_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
        required_profiles=["coding-default"],
        policy=Phase8ValidationPolicy(min_completed_cycles=2, min_promoted_cycles=1, min_best_overall_delta=0.05),
    )

    assert report["pass"] is True
    assert report["failed_checks"] == []
    coding = report["profiles"]["coding-default"]
    assert coding["summary"]["cycle_count"] == 2
    assert coding["summary"]["promoted_cycle_count"] == 1
    assert coding["summary"]["rollback_cycle_count"] == 1
    assert coding["checks"]["rollback_coverage"]["pass"] is True
    assert coding["checks"]["best_overall_delta"]["pass"] is True


def test_build_phase8_validation_report_fails_without_rollback_evidence_and_with_failed_jobs(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        experiment_id="exp-b",
        status="active",
        artifact_bytes=b"adapter-b",
        benchmark_summary={"overall_score": 0.74},
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-b",
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        overall_delta=0.19,
        recall_delta=0.12,
        promote=True,
        rollback=False,
    )
    failed = queue.enqueue_job(
        job_type="eval-profile",
        payload={
            "advisor_profile_id": "coding-default",
            "candidate_checkpoint_id": "ckpt-active",
            "benchmark_manifests": [],
            "promotion_threshold": 0.05,
        },
        resume_token="forced-eval:failed",
    )
    queue.update_job(failed.job_id, status="failed", last_error="boom")

    report = build_phase8_validation_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
        required_profiles=["coding-default"],
        policy=Phase8ValidationPolicy(min_completed_cycles=1, min_promoted_cycles=1, min_best_overall_delta=0.05),
    )

    assert report["pass"] is False
    assert "failed_jobs" in report["failed_checks"]
    coding = report["profiles"]["coding-default"]
    assert coding["checks"]["rollback_coverage"]["pass"] is False
    assert report["job_summary"]["failed"] == 1


def test_build_phase8_validation_report_counts_queued_jobs_in_job_summary(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    queue.enqueue_job(
        job_type="eval-profile",
        payload={
            "advisor_profile_id": "coding-default",
            "candidate_checkpoint_id": "ckpt-queued",
            "benchmark_manifests": [],
            "promotion_threshold": 0.05,
        },
        resume_token="continuous:exp-queued:coding-default:eval:ckpt-queued",
    )

    report = build_phase8_validation_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
        required_profiles=[],
        policy=Phase8ValidationPolicy(require_active_checkpoint=False, require_rollback_coverage=False),
    )

    assert report["job_summary"]["total"] == 1
    assert report["job_summary"]["queued"] == 1



def test_build_phase8_validation_report_counts_lineage_rollbacks_without_trend_history_evidence(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-rollback",
        experiment_id="exp-a",
        status="rolled_back",
        artifact_bytes=b"adapter-a",
        benchmark_summary={"overall_score": 0.47},
        rollback_reason="manual rollback",
    )
    _write_checkpoint(
        manager,
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        experiment_id="exp-b",
        status="active",
        artifact_bytes=b"adapter-b",
        benchmark_summary={"overall_score": 0.74},
    )
    _enqueue_cycle_jobs(
        queue,
        experiment_id="exp-b",
        profile_id="coding-default",
        checkpoint_id="ckpt-active",
        overall_delta=0.19,
        recall_delta=0.12,
        promote=True,
        rollback=False,
    )

    report = build_phase8_validation_report(
        lifecycle_manager=manager,
        job_records=queue.list_jobs(),
        required_profiles=["coding-default"],
        policy=Phase8ValidationPolicy(min_completed_cycles=1, min_promoted_cycles=1, min_best_overall_delta=0.05),
    )

    coding = report["profiles"]["coding-default"]
    assert coding["summary"]["rollback_cycle_count"] == 1
    assert coding["checks"]["rollback_coverage"]["pass"] is True
