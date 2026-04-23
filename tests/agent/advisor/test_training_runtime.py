import json
from pathlib import Path

from agent.advisor import ProfileCheckpointEvaluation, evaluate_profile_checkpoint_for_promotion
from agent.advisor.evaluation.benchmark import BenchmarkRunManifest
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.training.training_backends import GRPOTrainingBackend
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult
from agent.advisor.training.training_runtime import (
    CheckpointLifecycleManager,
    TrainingCheckpointRecord,
    TrainingJobConfig,
    TrainingJobResult,
    evaluate_trained_checkpoint,
    resolve_active_profile_checkpoint_metadata,
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


class StubTrainer:
    def train(self, request, checkpoint_dir, training_samples):
        adapter_path = checkpoint_dir / "adapters.safetensors"
        adapter_path.write_bytes(b"stub-adapter")
        config_path = checkpoint_dir / "adapter_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "base_model_name": request.training_config.base_model_name,
                    "adapter_method": request.training_config.adapter_method,
                    "lora_rank": request.training_config.lora_rank,
                    "target_modules": request.training_config.target_modules,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "artifact_paths": {
                "adapter_model": str(adapter_path),
                "adapter_config": str(config_path),
            },
            "metrics": {
                "train_loss": 0.11,
                "optimizer_steps": 3,
                "trained_examples": len(training_samples),
            },
        }


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



def test_resolve_active_profile_checkpoint_metadata_returns_manifest_and_artifacts(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "coding-default" / "coding-active"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapters.safetensors").write_bytes(b"adapter")
    checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
    checkpoint_manifest_path.write_text(
        json.dumps(
            {
                "checkpoint_id": "coding-active",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {
                    "adapter_model": str(checkpoint_dir / "adapters.safetensors"),
                    "adapter_config": str(checkpoint_dir / "adapter_config.json"),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-active",
            experiment_id="exp-14",
            path=str(checkpoint_dir),
            status="active",
            advisor_profile_id="coding-default",
        )
    )

    metadata = resolve_active_profile_checkpoint_metadata(
        advisor_profile_id="coding-default",
        lifecycle_manager=manager,
    )

    assert metadata["checkpoint_id"] == "coding-active"
    assert metadata["checkpoint_path"] == str(checkpoint_dir)
    assert metadata["manifest_path"] == str(checkpoint_manifest_path)
    assert metadata["artifact_paths"]["adapter_model"] == str(checkpoint_dir / "adapters.safetensors")



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
        backend=GRPOTrainingBackend(trainer=StubTrainer()),
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((tmp_path / "artifacts" / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert result.advisor_profile_id == "coding-default"
    assert result.backend_name == "grpo"
    assert manifest["advisor_profile_id"] == "coding-default"
    assert manifest["backend_name"] == "grpo"
    assert manifest["rollout_group_id"] == "group-train"
    assert manifest["backend_artifact_paths"]["adapter_model"].endswith("adapters.safetensors")
    assert manifest["backend_artifact_paths"]["adapter_config"].endswith("adapter_config.json")
    assert checkpoint_registry[0]["advisor_profile_id"] == "coding-default"
    assert checkpoint_registry[0]["status"] == "candidate"
    assert checkpoint_registry[0]["checkpoint_id"] == result.checkpoint_id



def test_run_profile_training_job_supports_researcher_profile(tmp_path):
    registry = AdvisorProfileRegistry.from_toml("config/advisor_profiles.toml")
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    rollout_group = TrainingRolloutGroupResult(
        group_id="group-research",
        advisor_profile_id="researcher",
        results=[
            TrainingRolloutResult(
                rollout_id="rollout-research-1",
                advisor_profile_id="researcher",
                packet={"run_id": "run-research-1", "task": {"domain": "research-writing"}},
                primary_advice={"recommended_plan": ["review sources", "draft structured summary"]},
                executor_result={"status": "success"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.91},
                diagnostics={"multi_turn": False},
            )
        ],
        reward_values=[0.91],
        summary={"mean_reward": 0.91},
    )

    result = run_profile_training_job(
        job_id="job-research",
        experiment_id="exp-14-research",
        advisor_profile_id="researcher",
        rollout_group=rollout_group,
        profile_registry=registry,
        lifecycle_manager=manager,
        backend=GRPOTrainingBackend(trainer=StubTrainer()),
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((tmp_path / "artifacts" / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert result.advisor_profile_id == "researcher"
    assert result.backend_name == "grpo"
    assert manifest["advisor_profile_id"] == "researcher"
    assert manifest["rollout_group_id"] == "group-research"
    assert checkpoint_registry[0]["advisor_profile_id"] == "researcher"
    assert checkpoint_registry[0]["checkpoint_id"] == result.checkpoint_id
    assert "backend_manifest" in result.backend_artifact_paths



def test_run_profile_training_job_supports_text_ui_profile(tmp_path):
    registry = AdvisorProfileRegistry.from_toml("config/advisor_profiles.toml")
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    rollout_group = TrainingRolloutGroupResult(
        group_id="group-text-ui",
        advisor_profile_id="text-ui",
        results=[
            TrainingRolloutResult(
                rollout_id="rollout-text-ui-1",
                advisor_profile_id="text-ui",
                packet={"run_id": "run-text-ui-1", "task": {"domain": "text-ui"}},
                primary_advice={"recommended_plan": ["review layout brief", "update layout spec"]},
                executor_result={"status": "success"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.88},
                diagnostics={"multi_turn": False},
            )
        ],
        reward_values=[0.88],
        summary={"mean_reward": 0.88},
    )

    result = run_profile_training_job(
        job_id="job-text-ui",
        experiment_id="exp-14-text-ui",
        advisor_profile_id="text-ui",
        rollout_group=rollout_group,
        profile_registry=registry,
        lifecycle_manager=manager,
        backend=GRPOTrainingBackend(trainer=StubTrainer()),
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((tmp_path / "artifacts" / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert result.advisor_profile_id == "text-ui"
    assert manifest["advisor_profile_id"] == "text-ui"
    assert manifest["rollout_group_id"] == "group-text-ui"
    assert checkpoint_registry[0]["advisor_profile_id"] == "text-ui"



def test_checkpoint_lifecycle_manager_filters_active_checkpoints_by_profile(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-active",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-active"),
            status="active",
            advisor_profile_id="coding-default",
        )
    )
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="ui-active",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "ui-active"),
            status="active",
            advisor_profile_id="image-ui",
        )
    )

    coding_active = manager.get_active_checkpoint("coding-default")
    ui_active = manager.get_active_checkpoint("image-ui")
    coding_records = manager.list_checkpoints(advisor_profile_id="coding-default")

    assert coding_active is not None
    assert coding_active.checkpoint_id == "coding-active"
    assert ui_active is not None
    assert ui_active.checkpoint_id == "ui-active"
    assert [record.checkpoint_id for record in coding_records] == ["coding-active"]


def test_evaluate_profile_checkpoint_for_promotion_promotes_passing_candidate(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-active",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-active"),
            status="active",
            advisor_profile_id="coding-default",
        )
    )
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-candidate",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-candidate"),
            status="candidate",
            advisor_profile_id="coding-default",
        )
    )
    manifests = [
        _benchmark_manifest("baseline-run", arm="baseline", overall_score=0.55).model_copy(
            update={"advisor_profile_id": "coding-default"}
        ),
        _benchmark_manifest("candidate-run", arm="advisor", overall_score=0.82).model_copy(
            update={"advisor_profile_id": "coding-default"}
        ),
        _benchmark_manifest("other-profile-run", arm="advisor", overall_score=0.99).model_copy(
            update={"advisor_profile_id": "image-ui"}
        ),
    ]

    evaluation = evaluate_profile_checkpoint_for_promotion(
        advisor_profile_id="coding-default",
        candidate_checkpoint_id="coding-candidate",
        benchmark_manifests=manifests,
        lifecycle_manager=manager,
        promotion_threshold=0.1,
    )

    assert isinstance(evaluation, ProfileCheckpointEvaluation)
    assert evaluation.promote is True
    assert evaluation.rollback is False
    assert evaluation.active_checkpoint_id == "coding-active"
    assert evaluation.benchmark_manifest_count == 2
    assert manager.get_active_checkpoint("coding-default").checkpoint_id == "coding-candidate"


def test_evaluate_profile_checkpoint_for_promotion_rolls_back_failing_candidate(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-active",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-active"),
            status="active",
            advisor_profile_id="coding-default",
        )
    )
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-candidate",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-candidate"),
            status="candidate",
            advisor_profile_id="coding-default",
        )
    )
    manifests = [
        _benchmark_manifest("baseline-run", arm="baseline", overall_score=0.85).model_copy(
            update={"advisor_profile_id": "coding-default"}
        ),
        _benchmark_manifest("candidate-run", arm="advisor", overall_score=0.62).model_copy(
            update={"advisor_profile_id": "coding-default"}
        ),
    ]

    evaluation = evaluate_profile_checkpoint_for_promotion(
        advisor_profile_id="coding-default",
        candidate_checkpoint_id="coding-candidate",
        benchmark_manifests=manifests,
        lifecycle_manager=manager,
        promotion_threshold=0.1,
    )

    candidate_record = manager.get_checkpoint("coding-candidate")
    assert evaluation.promote is False
    assert evaluation.rollback is True
    assert "coding-default" in evaluation.decision_reason
    assert candidate_record is not None
    assert candidate_record.status == "rolled_back"
    assert candidate_record.rollback_reason == evaluation.decision_reason


def test_evaluate_profile_checkpoint_for_promotion_rejects_missing_profile_manifests(tmp_path):
    manager = CheckpointLifecycleManager(tmp_path / "artifacts")
    manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="coding-candidate",
            experiment_id="exp-14",
            path=str(tmp_path / "artifacts" / "checkpoints" / "coding-candidate"),
            status="candidate",
            advisor_profile_id="coding-default",
        )
    )

    try:
        evaluate_profile_checkpoint_for_promotion(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="coding-candidate",
            benchmark_manifests=[
                _benchmark_manifest("other-profile-run", arm="advisor", overall_score=0.91).model_copy(
                    update={"advisor_profile_id": "image-ui"}
                )
            ],
            lifecycle_manager=manager,
            promotion_threshold=0.1,
        )
    except ValueError as exc:
        assert "benchmark manifests" in str(exc)
    else:
        raise AssertionError("expected missing profile-local manifests to raise ValueError")


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
