from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .profiles import AdvisorTrainingConfig
from .training_rollouts import TrainingRolloutGroupResult


class TrainingBackendRunRequest(BaseModel):
    job_id: str
    experiment_id: str
    advisor_profile_id: str
    training_config: AdvisorTrainingConfig
    rollout_group: TrainingRolloutGroupResult
    output_dir: str
    base_checkpoint_path: str | None = None


class TrainingBackendRunResult(BaseModel):
    job_id: str
    experiment_id: str
    advisor_profile_id: str
    backend_name: str
    checkpoint_id: str
    checkpoint_path: str
    training_metrics: dict = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    rollout_summary: dict = Field(default_factory=dict)


class GRPOTrainingBackend:
    backend_name = "grpo"

    def run(self, request: TrainingBackendRunRequest) -> TrainingBackendRunResult:
        output_root = Path(request.output_dir)
        job_dir = output_root / "training-jobs" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_id = f"{request.advisor_profile_id}-{request.job_id}"
        checkpoint_root = Path(request.training_config.checkpoint_root)
        checkpoint_dir = checkpoint_root if checkpoint_root.is_absolute() else output_root / checkpoint_root
        checkpoint_dir = checkpoint_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        reward_values = [float(value) for value in request.rollout_group.reward_values]
        rollout_count = request.rollout_group.rollout_count
        mean_reward = round(sum(reward_values) / rollout_count, 4) if reward_values else 0.0
        training_metrics = {
            "mean_reward": mean_reward,
            "max_reward": max(reward_values) if reward_values else 0.0,
            "min_reward": min(reward_values) if reward_values else 0.0,
            "rollout_count": rollout_count,
            "num_generations": request.training_config.num_generations,
            "max_steps": request.training_config.max_steps,
        }

        checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
        checkpoint_manifest_path.write_text(
            json.dumps(
                {
                    "checkpoint_id": checkpoint_id,
                    "job_id": request.job_id,
                    "experiment_id": request.experiment_id,
                    "advisor_profile_id": request.advisor_profile_id,
                    "backend_name": self.backend_name,
                    "training_metrics": training_metrics,
                    "rollout_summary": request.rollout_group.summary,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        backend_manifest_path = job_dir / "backend-manifest.json"
        backend_manifest_path.write_text(
            json.dumps(
                {
                    "job_id": request.job_id,
                    "experiment_id": request.experiment_id,
                    "advisor_profile_id": request.advisor_profile_id,
                    "backend_name": self.backend_name,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_path": str(checkpoint_dir),
                    "training_config": request.training_config.model_dump(),
                    "rollout_group_id": request.rollout_group.group_id,
                    "rollout_summary": request.rollout_group.summary,
                    "training_metrics": training_metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return TrainingBackendRunResult(
            job_id=request.job_id,
            experiment_id=request.experiment_id,
            advisor_profile_id=request.advisor_profile_id,
            backend_name=self.backend_name,
            checkpoint_id=checkpoint_id,
            checkpoint_path=str(checkpoint_dir),
            training_metrics=training_metrics,
            artifact_paths={
                "backend_manifest": str(backend_manifest_path),
                "checkpoint_manifest": str(checkpoint_manifest_path),
            },
            rollout_summary=request.rollout_group.summary,
        )
