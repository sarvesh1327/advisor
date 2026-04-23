from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .profiles import AdvisorProfileRegistry
from .training_backends import GRPOTrainingBackend
from .training_rollouts import TrainingRolloutGroupResult


class TrainingJobConfig(BaseModel):
    experiment_id: str
    training_mode: str
    dataset_manifest: dict = Field(default_factory=dict)
    benchmark_manifests: list[dict] = Field(default_factory=list)
    output_dir: str
    advisor_profile_id: str | None = None
    backend_name: str | None = None
    rollout_group_id: str | None = None
    rollout_group_path: str | None = None
    backend_artifact_paths: dict[str, str] = Field(default_factory=dict)


class TrainingJobResult(BaseModel):
    job_id: str
    experiment_id: str
    checkpoint_id: str
    manifest_path: str
    artifact_dir: str
    training_metrics: dict = Field(default_factory=dict)
    advisor_profile_id: str | None = None
    backend_name: str | None = None
    rollout_group_id: str | None = None
    backend_artifact_paths: dict[str, str] = Field(default_factory=dict)


class TrainingCheckpointRecord(BaseModel):
    checkpoint_id: str
    experiment_id: str
    path: str
    status: str
    benchmark_summary: dict = Field(default_factory=dict)
    rollback_reason: str | None = None
    advisor_profile_id: str | None = None


class CheckpointLifecycleManager:
    def __init__(self, artifacts_root: str | Path):
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.artifacts_root / "checkpoint_registry.json"

    def register_checkpoint(self, record: TrainingCheckpointRecord) -> TrainingCheckpointRecord:
        registry = self._load_registry()
        registry = [item for item in registry if item.get("checkpoint_id") != record.checkpoint_id]
        registry.append(record.model_dump())
        self._write_registry(registry)
        Path(record.path).mkdir(parents=True, exist_ok=True)
        return record

    def promote_checkpoint(self, checkpoint_id: str) -> TrainingCheckpointRecord:
        registry = self._load_registry()
        updated = []
        active_record = None
        for item in registry:
            if item["checkpoint_id"] == checkpoint_id:
                item["status"] = "active"
                active_record = TrainingCheckpointRecord.model_validate(item)
            elif item.get("status") == "active":
                item["status"] = "candidate"
            updated.append(item)
        self._write_registry(updated)
        if active_record is None:
            raise ValueError(f"unknown checkpoint_id: {checkpoint_id}")
        return active_record

    def rollback_to_checkpoint(self, checkpoint_id: str, *, reason: str) -> TrainingCheckpointRecord:
        registry = self._load_registry()
        record = None
        for item in registry:
            if item["checkpoint_id"] == checkpoint_id:
                item["status"] = "rolled_back"
                item["rollback_reason"] = reason
                record = TrainingCheckpointRecord.model_validate(item)
                break
        self._write_registry(registry)
        if record is None:
            raise ValueError(f"unknown checkpoint_id: {checkpoint_id}")
        return record

    def record_training_job(self, result: TrainingJobResult, *, config: TrainingJobConfig) -> TrainingJobResult:
        artifact_dir = self.artifacts_root / "training-jobs" / result.job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "training-manifest.json"
        manifest = {
            "job_id": result.job_id,
            "experiment_id": config.experiment_id,
            "training_mode": config.training_mode,
            "dataset_manifest": config.dataset_manifest,
            "benchmark_manifests": config.benchmark_manifests,
            "training_metrics": result.training_metrics,
            "checkpoint_id": result.checkpoint_id,
            "advisor_profile_id": config.advisor_profile_id,
            "backend_name": config.backend_name,
            "rollout_group_id": config.rollout_group_id,
            "rollout_group_path": config.rollout_group_path,
            "backend_artifact_paths": config.backend_artifact_paths,
            "created_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return result.model_copy(update={"manifest_path": str(manifest_path), "artifact_dir": str(artifact_dir)})

    def record_rollout_group(self, group: TrainingRolloutGroupResult, *, job_id: str) -> str:
        artifact_dir = self.artifacts_root / "training-jobs" / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / "rollout-group.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "group_id": group.group_id,
                    "advisor_profile_id": group.advisor_profile_id,
                    "rollout_count": group.rollout_count,
                    "reward_values": group.reward_values,
                    "summary": group.summary,
                    "results": [result.model_dump() for result in group.results],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return str(manifest_path)

    def _load_registry(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, registry: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def run_profile_training_job(
    *,
    job_id: str,
    experiment_id: str,
    advisor_profile_id: str,
    rollout_group: TrainingRolloutGroupResult,
    profile_registry: AdvisorProfileRegistry,
    lifecycle_manager: CheckpointLifecycleManager,
    backend: GRPOTrainingBackend | None = None,
) -> TrainingJobResult:
    profile = profile_registry.resolve(advisor_profile_id)
    training_config = profile.training
    if training_config is None:
        raise ValueError(f"advisor profile {advisor_profile_id} is missing training config")

    rollout_group_path = lifecycle_manager.record_rollout_group(rollout_group, job_id=job_id)
    selected_backend = backend or GRPOTrainingBackend()
    backend_result = selected_backend.run(
        request=selected_backend_request(
            job_id=job_id,
            experiment_id=experiment_id,
            advisor_profile_id=advisor_profile_id,
            training_config=training_config,
            rollout_group=rollout_group,
            output_dir=str(lifecycle_manager.artifacts_root),
        )
    )
    lifecycle_manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id=backend_result.checkpoint_id,
            experiment_id=experiment_id,
            path=backend_result.checkpoint_path,
            status="candidate",
            benchmark_summary={},
            advisor_profile_id=advisor_profile_id,
        )
    )
    config = TrainingJobConfig(
        experiment_id=experiment_id,
        training_mode=selected_backend.backend_name,
        dataset_manifest={},
        benchmark_manifests=[],
        output_dir=str(lifecycle_manager.artifacts_root),
        advisor_profile_id=advisor_profile_id,
        backend_name=backend_result.backend_name,
        rollout_group_id=rollout_group.group_id,
        rollout_group_path=rollout_group_path,
        backend_artifact_paths=backend_result.artifact_paths,
    )
    result = TrainingJobResult(
        job_id=job_id,
        experiment_id=experiment_id,
        checkpoint_id=backend_result.checkpoint_id,
        manifest_path="",
        artifact_dir="",
        training_metrics=backend_result.training_metrics,
        advisor_profile_id=advisor_profile_id,
        backend_name=backend_result.backend_name,
        rollout_group_id=rollout_group.group_id,
        backend_artifact_paths=backend_result.artifact_paths,
    )
    return lifecycle_manager.record_training_job(result, config=config)


def selected_backend_request(
    *,
    job_id: str,
    experiment_id: str,
    advisor_profile_id: str,
    training_config,
    rollout_group: TrainingRolloutGroupResult,
    output_dir: str,
):
    from .training_backends import TrainingBackendRunRequest

    return TrainingBackendRunRequest(
        job_id=job_id,
        experiment_id=experiment_id,
        advisor_profile_id=advisor_profile_id,
        training_config=training_config,
        rollout_group=rollout_group,
        output_dir=output_dir,
    )


def evaluate_trained_checkpoint(
    *,
    checkpoint_id: str,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
    promotion_threshold: float = 0.05,
) -> dict:
    overall_delta = round(float(candidate_summary.get("overall_score", 0.0)) - float(baseline_summary.get("overall_score", 0.0)), 4)
    recall_delta = round(
        float(candidate_summary.get("focus_target_recall", 0.0)) - float(baseline_summary.get("focus_target_recall", 0.0)),
        4,
    )
    promote = overall_delta >= promotion_threshold
    rollback = overall_delta < 0.0 or recall_delta < 0.0
    return {
        "checkpoint_id": checkpoint_id,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "deltas": {
            "overall_score": overall_delta,
            "focus_target_recall": recall_delta,
        },
        "promote": promote,
        "rollback": rollback,
    }
