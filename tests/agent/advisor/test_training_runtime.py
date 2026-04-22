import json
from pathlib import Path

from agent.advisor.benchmark import BenchmarkRunManifest
from agent.advisor.training_runtime import (
    CheckpointLifecycleManager,
    TrainingCheckpointRecord,
    TrainingJobConfig,
    TrainingJobResult,
    evaluate_trained_checkpoint,
)


def _benchmark_manifest(run_id: str, arm: str = "advisor", overall_score: float = 0.8):
    return BenchmarkRunManifest(
        run_id=run_id,
        fixture_id=f"fixture-{run_id}",
        domain="coding",
        split="validation",
        packet_hash=f"hash-{run_id}",
        executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
        verifier_set=["build-check"],
        routing_arm=arm,
        reward_version="phase8-v1",
        score={"overall_score": overall_score, "focus_target_recall": overall_score},
    )


def test_checkpoint_lifecycle_manager_registers_promotes_and_rolls_back(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    candidate = manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="ckpt-1",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "ckpt-1"),
            status="candidate",
            benchmark_summary={"overall_score": 0.82},
        )
    )

    active = manager.promote_checkpoint("ckpt-1")
    rolled_back = manager.rollback_to_checkpoint("ckpt-1", reason="regression detected")
    registry = json.loads((tmp_path / "artifacts" / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert candidate.status == "candidate"
    assert active.status == "active"
    assert rolled_back.status == "rolled_back"
    assert registry[0]["rollback_reason"] == "regression detected"


def test_training_job_result_persists_manifest_and_artifacts(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    config = TrainingJobConfig(
        experiment_id="exp-14",
        training_mode="supervised",
        dataset_manifest={"examples": [{"run_id": "run-a"}]},
        benchmark_manifests=[_benchmark_manifest("run-a").model_dump()],
        output_dir=str(tmp_path / "artifacts"),
    )

    result = manager.record_training_job(
        TrainingJobResult(
            job_id="job-1",
            experiment_id="exp-14",
            checkpoint_id="ckpt-1",
            manifest_path="",
            artifact_dir="",
            training_metrics={"loss": 0.12},
        ),
        config=config,
    )

    manifest_path = Path(result.manifest_path)
    artifact_dir = Path(result.artifact_dir)

    assert manifest_path.exists()
    assert artifact_dir.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "exp-14"
    assert manifest["training_mode"] == "supervised"
    assert manifest["dataset_manifest"]["examples"][0]["run_id"] == "run-a"


def test_evaluate_trained_checkpoint_reports_promotion_decision_and_regression_signals():
    report = evaluate_trained_checkpoint(
        checkpoint_id="ckpt-2",
        baseline_summary={"overall_score": 0.61, "focus_target_recall": 0.55},
        candidate_summary={"overall_score": 0.84, "focus_target_recall": 0.8},
        promotion_threshold=0.1,
    )
    regression = evaluate_trained_checkpoint(
        checkpoint_id="ckpt-3",
        baseline_summary={"overall_score": 0.8, "focus_target_recall": 0.8},
        candidate_summary={"overall_score": 0.6, "focus_target_recall": 0.5},
        promotion_threshold=0.1,
    )

    assert report["checkpoint_id"] == "ckpt-2"
    assert report["promote"] is True
    assert report["deltas"]["overall_score"] == 0.23
    assert regression["promote"] is False
    assert regression["rollback"] is True
