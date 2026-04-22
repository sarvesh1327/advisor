from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class TrainingJobConfig(BaseModel):
    experiment_id: str
    training_mode: str
    dataset_manifest: dict = Field(default_factory=dict)
    benchmark_manifests: list[dict] = Field(default_factory=list)
    output_dir: str


class TrainingJobResult(BaseModel):
    job_id: str
    experiment_id: str
    checkpoint_id: str
    manifest_path: str
    artifact_dir: str
    training_metrics: dict = Field(default_factory=dict)


class TrainingCheckpointRecord(BaseModel):
    checkpoint_id: str
    experiment_id: str
    path: str
    status: str
    benchmark_summary: dict = Field(default_factory=dict)
    rollback_reason: str | None = None


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
            "created_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return result.model_copy(update={"manifest_path": str(manifest_path), "artifact_dir": str(artifact_dir)})

    def _load_registry(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, registry: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


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
