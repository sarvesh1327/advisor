import json
from pathlib import Path

from agent.advisor.benchmark import BenchmarkRunManifest
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult
from agent.advisor.training_runtime import (
    CheckpointLifecycleManager,
    TrainingCheckpointRecord,
    TrainingJobConfig,
    TrainingJobResult,
    evaluate_trained_checkpoint,
    run_profile_training_job,
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


def test_checkpoint_lifecycle_manager_persists_rollout_group_manifest(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    group = TrainingRolloutGroupResult(
        group_id="group-1",
        advisor_profile_id="coding-default",
        results=[
            TrainingRolloutResult(
                rollout_id="rollout-1",
                advisor_profile_id="coding-default",
                packet={"run_id": "run-1"},
                primary_advice={"recommended_plan": ["inspect main.py"]},
                executor_result={"status": "success"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.9},
                diagnostics={"multi_turn": False},
            )
        ],
        reward_values=[0.9],
        summary={"mean_reward": 0.9},
    )

    manifest_path = Path(manager.record_rollout_group(group, job_id="job-rollout"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert manifest["group_id"] == "group-1"
    assert manifest["advisor_profile_id"] == "coding-default"
    assert manifest["rollout_count"] == 1
    assert manifest["reward_values"] == [0.9]


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


def test_run_profile_training_job_records_profile_owned_checkpoint_and_manifest(tmp_path):
    registry = AdvisorProfileRegistry.from_toml("config/advisor_profiles.toml")
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    rollout_group = TrainingRolloutGroupResult(
        group_id="group-train",
        advisor_profile_id="coding-default",
        results=[
            TrainingRolloutResult(
                rollout_id="rollout-1",
                advisor_profile_id="coding-default",
                packet={"run_id": "run-1"},
                primary_advice={"recommended_plan": ["inspect main.py"]},
                executor_result={"status": "success"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.8},
                diagnostics={"multi_turn": False},
            )
        ],
        reward_values=[0.8],
        summary={"mean_reward": 0.8},
    )

    result = run_profile_training_job(
        job_id="job-2",
        experiment_id="exp-14",
        advisor_profile_id="coding-default",
        rollout_group=rollout_group,
        profile_registry=registry,
        lifecycle_manager=manager,
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((tmp_path / "artifacts" / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert result.advisor_profile_id == "coding-default"
    assert result.backend_name == "grpo"
    assert manifest["advisor_profile_id"] == "coding-default"
    assert manifest["backend_name"] == "grpo"
    assert manifest["rollout_group_id"] == "group-train"
    assert checkpoint_registry[0]["advisor_profile_id"] == "coding-default"
    assert checkpoint_registry[0]["status"] == "candidate"


def test_run_profile_training_job_rejects_profiles_without_training_config(tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "\n".join(
            [
                'default_profile_id = "coding-default"',
                "",
                "[profiles.coding-default]",
                'domain = "coding"',
            ]
        ),
        encoding="utf-8",
    )
    registry = AdvisorProfileRegistry.from_toml(profiles_path)
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    rollout_group = TrainingRolloutGroupResult(
        group_id="group-train",
        advisor_profile_id="coding-default",
        results=[],
        reward_values=[],
        summary={},
    )

    try:
        run_profile_training_job(
            job_id="job-3",
            experiment_id="exp-14",
            advisor_profile_id="coding-default",
            rollout_group=rollout_group,
            profile_registry=registry,
            lifecycle_manager=manager,
        )
    except ValueError as exc:
        assert "training config" in str(exc)
    else:
        raise AssertionError("expected missing training config to raise ValueError")
