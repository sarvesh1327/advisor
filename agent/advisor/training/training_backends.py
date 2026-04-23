from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult

try:
    import mlx.optimizers as mlx_optim
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner import TrainingArgs as mlx_TrainingArgs
    from mlx_lm.tuner import linear_to_lora_layers as mlx_linear_to_lora_layers
    from mlx_lm.tuner import train as mlx_train
    from mlx_lm.tuner.callbacks import TrainingCallback as MLXTrainingCallback
    from mlx_lm.tuner.datasets import CompletionsDataset as MLXCompletionsDataset
    from mlx_lm.utils import save_config as mlx_save_config
except ImportError:
    mlx_optim = None
    mlx_lm_load = None
    mlx_TrainingArgs = None
    mlx_linear_to_lora_layers = None
    mlx_train = None
    MLXTrainingCallback = object
    MLXCompletionsDataset = None
    mlx_save_config = None


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


class GRPOTrainingSample(BaseModel):
    prompt: str
    completion: str
    reward: float
    profile_id: str
    provenance: dict[str, str | int | None] = Field(default_factory=dict)


class TrainerRunArtifact(BaseModel):
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)


class _MLXTrainingReporter(MLXTrainingCallback):
    def __init__(self):
        self.last_train_info: dict[str, Any] = {}
        self.last_val_info: dict[str, Any] = {}

    def on_train_loss_report(self, train_info: dict):
        self.last_train_info = dict(train_info)

    def on_val_loss_report(self, val_info: dict):
        self.last_val_info = dict(val_info)


class MLXLoRATrainer:
    def train(
        self,
        request: TrainingBackendRunRequest,
        checkpoint_dir: Path,
        training_samples: list[GRPOTrainingSample],
    ) -> TrainerRunArtifact:
        _ensure_mlx_training_dependencies()
        if not training_samples:
            raise ValueError("training backend requires at least one GRPO training sample")
        if request.training_config.adapter_method != "lora":
            raise ValueError(
                f"unsupported adapter_method for MLX trainer: {request.training_config.adapter_method}"
            )

        base_model_name = request.training_config.base_model_name
        if not base_model_name:
            raise ValueError("base_model_name is required for real LoRA training")

        model, tokenizer = mlx_lm_load(base_model_name)
        if not hasattr(model, "layers"):
            raise ValueError("loaded MLX model does not expose transformer layers for LoRA training")

        model.freeze()
        target_modules = list(request.training_config.target_modules)
        lora_parameters = {
            "rank": int(request.training_config.lora_rank or 0),
            "scale": float(request.training_config.lora_alpha or request.training_config.lora_rank or 1),
            "dropout": float(request.training_config.lora_dropout),
            "keys": target_modules,
        }
        num_layers = len(model.layers)
        mlx_linear_to_lora_layers(model, num_layers, lora_parameters)

        dataset_rows = [
            {"prompt": sample.prompt, "completion": sample.completion}
            for sample in training_samples
        ]
        train_dataset = MLXCompletionsDataset(
            data=dataset_rows,
            tokenizer=tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=True,
        )

        adapter_path = checkpoint_dir / "adapters.safetensors"
        reporter = _MLXTrainingReporter()
        batch_size = min(max(1, len(training_samples)), 4)
        max_seq_length = max(
            int(request.training_config.max_prompt_tokens + request.training_config.max_completion_tokens),
            256,
        )
        training_args = mlx_TrainingArgs(
            batch_size=batch_size,
            iters=max(1, int(request.training_config.max_steps)),
            steps_per_report=1,
            steps_per_eval=max(2, int(request.training_config.max_steps) + 1),
            steps_per_save=max(1, int(request.training_config.max_steps)),
            max_seq_length=max_seq_length,
            adapter_file=str(adapter_path),
            grad_accumulation_steps=1,
        )
        optimizer = mlx_optim.Adam(learning_rate=1e-5)
        mlx_train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=None,
            args=training_args,
            training_callback=reporter,
        )

        adapter_config_path = checkpoint_dir / "adapter_config.json"
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "lora_parameters": lora_parameters,
            "base_model_name": base_model_name,
            "adapter_method": request.training_config.adapter_method,
            "lora_rank": request.training_config.lora_rank,
            "lora_alpha": request.training_config.lora_alpha,
            "lora_dropout": request.training_config.lora_dropout,
            "target_modules": target_modules,
            "advisor_profile_id": request.advisor_profile_id,
            "job_id": request.job_id,
            "experiment_id": request.experiment_id,
        }
        mlx_save_config(adapter_config, adapter_config_path)

        train_metrics = {
            "train_loss": round(float(reporter.last_train_info.get("train_loss", 0.0)), 6),
            "optimizer_steps": int(reporter.last_train_info.get("iteration", 0)),
            "trained_examples": len(training_samples),
        }
        if reporter.last_train_info.get("trained_tokens") is not None:
            train_metrics["trained_tokens"] = int(reporter.last_train_info["trained_tokens"])
        if reporter.last_train_info.get("tokens_per_second") is not None:
            train_metrics["tokens_per_second"] = round(
                float(reporter.last_train_info["tokens_per_second"]),
                4,
            )

        return TrainerRunArtifact(
            artifact_paths={
                "adapter_model": str(adapter_path),
                "adapter_config": str(adapter_config_path),
            },
            metrics=train_metrics,
        )


def build_grpo_training_samples(request: TrainingBackendRunRequest) -> list[GRPOTrainingSample]:
    # Keep sample construction deterministic so reward provenance is replayable across runs.
    samples: list[GRPOTrainingSample] = []
    for sample_index, result in enumerate(request.rollout_group.results):
        packet_payload = _normalized_payload(result.packet)
        advice_payload = _normalized_payload(result.primary_advice)
        samples.append(
            GRPOTrainingSample(
                prompt=json.dumps(packet_payload, sort_keys=True),
                completion=json.dumps(advice_payload, sort_keys=True),
                reward=float(_reward_total(result.reward_label)),
                profile_id=request.advisor_profile_id,
                provenance={
                    "job_id": request.job_id,
                    "group_id": request.rollout_group.group_id,
                    "rollout_id": result.rollout_id,
                    "advisor_profile_id": result.advisor_profile_id,
                    "sample_index": sample_index,
                },
            )
        )
    return samples


class GRPOTrainingBackend:
    backend_name = "grpo"

    def __init__(self, trainer: Any | None = None):
        self.trainer = trainer or MLXLoRATrainer()

    def run(self, request: TrainingBackendRunRequest) -> TrainingBackendRunResult:
        output_root = Path(request.output_dir)
        job_dir = output_root / "training-jobs" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_id = f"{request.advisor_profile_id}-{request.job_id}"
        checkpoint_root = Path(request.training_config.checkpoint_root)
        checkpoint_dir = checkpoint_root if checkpoint_root.is_absolute() else output_root / checkpoint_root
        checkpoint_dir = checkpoint_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_samples = build_grpo_training_samples(request)
        reward_values = [float(value) for value in request.rollout_group.reward_values]
        rollout_count = request.rollout_group.rollout_count
        mean_reward = round(sum(reward_values) / rollout_count, 4) if reward_values else 0.0
        trainer_result = _normalize_trainer_result(
            self.trainer.train(request, checkpoint_dir, training_samples)
        )
        training_metrics = {
            "mean_reward": mean_reward,
            "max_reward": max(reward_values) if reward_values else 0.0,
            "min_reward": min(reward_values) if reward_values else 0.0,
            "rollout_count": rollout_count,
            "num_generations": request.training_config.num_generations,
            "max_steps": request.training_config.max_steps,
            "training_sample_count": len(training_samples),
            **trainer_result.metrics,
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
                    "artifact_paths": trainer_result.artifact_paths,
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
                    "training_sample_count": len(training_samples),
                    "artifact_paths": trainer_result.artifact_paths,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        artifact_paths = {
            "backend_manifest": str(backend_manifest_path),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            **trainer_result.artifact_paths,
        }
        return TrainingBackendRunResult(
            job_id=request.job_id,
            experiment_id=request.experiment_id,
            advisor_profile_id=request.advisor_profile_id,
            backend_name=self.backend_name,
            checkpoint_id=checkpoint_id,
            checkpoint_path=str(checkpoint_dir),
            training_metrics=training_metrics,
            artifact_paths=artifact_paths,
            rollout_summary=request.rollout_group.summary,
        )


def _normalized_payload(payload: BaseModel | dict) -> dict:
    # Normalize Pydantic models and dicts into a stable JSON shape before sample serialization.
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    return dict(payload)


def _reward_total(reward_label: BaseModel | dict) -> float:
    # Reward labels can still arrive as plain dicts in tests and persisted artifact paths.
    if isinstance(reward_label, BaseModel):
        return float(reward_label.model_dump().get("total_reward", 0.0))
    return float(dict(reward_label).get("total_reward", 0.0))


def _normalize_trainer_result(result: TrainerRunArtifact | dict) -> TrainerRunArtifact:
    if isinstance(result, TrainerRunArtifact):
        return result
    return TrainerRunArtifact.model_validate(result)


def _ensure_mlx_training_dependencies() -> None:
    if (
        mlx_optim is None
        or mlx_lm_load is None
        or mlx_TrainingArgs is None
        or mlx_linear_to_lora_layers is None
        or mlx_train is None
        or MLXCompletionsDataset is None
        or mlx_save_config is None
    ):
        raise RuntimeError(
            "mlx-lm training dependencies are unavailable; install advisor runtime extras to enable real LoRA training"
        )
