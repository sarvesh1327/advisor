from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as mlx_optim
    import numpy as np
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.tuner import TrainingArgs as mlx_TrainingArgs
    from mlx_lm.tuner import linear_to_lora_layers as mlx_linear_to_lora_layers
    from mlx_lm.tuner import train as mlx_train
    from mlx_lm.tuner.callbacks import TrainingCallback as MLXTrainingCallback
    from mlx_lm.tuner.datasets import CompletionsDataset as MLXCompletionsDataset
    from mlx_lm.utils import save_config as mlx_save_config
except ImportError:
    np = None
    mx = None
    nn = None
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
    provenance: dict[str, Any] = Field(default_factory=dict)


class GRPOTrainingCandidate(BaseModel):
    prompt: str
    completion: str
    reward: float
    advantage: float
    profile_id: str
    group_id: str
    candidate_index: int
    provenance: dict[str, Any] = Field(default_factory=dict)


class GRPOTrainingGroup(BaseModel):
    group_id: str
    profile_id: str
    candidates: list[GRPOTrainingCandidate] = Field(default_factory=list)
    reward_mean: float = 0.0
    reward_std: float = 0.0


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
        training_groups: list[GRPOTrainingGroup],
    ) -> TrainerRunArtifact:
        _ensure_mlx_training_dependencies()
        flattened_candidates = [candidate for group in training_groups for candidate in group.candidates]
        if not flattened_candidates:
            raise ValueError("training backend requires at least one GRPO training candidate")
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
            {"prompt": candidate.prompt, "completion": candidate.completion}
            for candidate in flattened_candidates
        ]
        train_dataset = MLXCompletionsDataset(
            data=dataset_rows,
            tokenizer=tokenizer,
            prompt_key="prompt",
            completion_key="completion",
            mask_prompt=True,
        )

        advantages = [float(candidate.advantage) for candidate in flattened_candidates]
        iterate_batches = _build_weighted_iterate_batches(advantages)
        adapter_path = checkpoint_dir / "adapters.safetensors"
        reporter = _MLXTrainingReporter()
        batch_size = min(max(1, len(flattened_candidates)), 4)
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
            loss=_grpo_weighted_loss,
            iterate_batches=iterate_batches,
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
            "trained_examples": len(flattened_candidates),
            "training_group_count": len(training_groups),
            "positive_advantage_count": sum(1 for value in advantages if value > 0),
            "negative_advantage_count": sum(1 for value in advantages if value < 0),
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
    seen_signatures: set[str] = set()
    for sample_index, result in enumerate(request.rollout_group.results):
        trajectory_payload = _normalized_payload(result.trajectory) if result.trajectory else {}
        trajectory_turns = list(trajectory_payload.get("turns") or [])
        if trajectory_turns:
            for turn_payload in sorted(
                (_normalized_payload(turn) for turn in trajectory_turns),
                key=lambda turn: int(turn.get("turn_index", 0)),
            ):
                sample = _trajectory_turn_training_sample(
                    request,
                    result,
                    trajectory_payload=trajectory_payload,
                    turn_payload=turn_payload,
                    sample_index=len(samples),
                )
                signature = _sample_signature(sample)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                samples.append(sample)
            continue

        packet_payload = _normalized_payload(result.packet)
        advice_payload = _normalized_payload(result.primary_advice)
        sample = GRPOTrainingSample(
            prompt=_stable_json_dumps(packet_payload),
            completion=_stable_json_dumps(advice_payload),
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
        signature = _sample_signature(sample)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        samples.append(sample)
    return samples


def _trajectory_turn_training_sample(
    request: TrainingBackendRunRequest,
    result: Any,
    *,
    trajectory_payload: dict[str, Any],
    turn_payload: dict[str, Any],
    sample_index: int,
) -> GRPOTrainingSample:
    final_reward = _trajectory_final_reward(trajectory_payload, result.reward_label)
    trajectory_id = trajectory_payload.get("trajectory_id") or result.rollout_id
    profile_id = trajectory_payload.get("advisor_profile_id") or result.advisor_profile_id
    turn_index = int(turn_payload.get("turn_index", sample_index))
    state_t = _normalized_payload(turn_payload.get("state_packet") or {})
    advice_t = _normalized_payload(turn_payload.get("advice") or {})
    observation_t = _normalized_payload(turn_payload.get("observation") or {})
    prompt_payload = {
        "state_t": state_t,
        "observation_t": observation_t,
        "turn_index": turn_index,
        "trajectory_id": trajectory_id,
        "final_reward": final_reward,
        "profile_id": profile_id,
    }
    completion_payload = {"advice_t": advice_t}
    return GRPOTrainingSample(
        prompt=_stable_json_dumps(prompt_payload),
        completion=_stable_json_dumps(completion_payload),
        reward=final_reward,
        profile_id=request.advisor_profile_id,
        provenance={
            "job_id": request.job_id,
            "group_id": request.rollout_group.group_id,
            "rollout_id": result.rollout_id,
            "advisor_profile_id": result.advisor_profile_id,
            "sample_index": sample_index,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
            "final_reward": final_reward,
            "profile_id": profile_id,
        },
    )


def build_grpo_training_groups(request: TrainingBackendRunRequest) -> list[GRPOTrainingGroup]:
    samples = build_grpo_training_samples(request)
    reward_values = [sample.reward for sample in samples]
    reward_mean = _rounded_reward_stat(sum(reward_values) / len(reward_values)) if reward_values else 0.0
    reward_std = _rounded_reward_stat(_population_stddev(reward_values)) if reward_values else 0.0
    candidates = [
        GRPOTrainingCandidate(
            prompt=sample.prompt,
            completion=sample.completion,
            reward=sample.reward,
            advantage=_reward_advantage(sample.reward, reward_mean, reward_std),
            profile_id=sample.profile_id,
            group_id=request.rollout_group.group_id,
            candidate_index=sample_index,
            provenance=sample.provenance,
        )
        for sample_index, sample in enumerate(samples)
    ]
    return [
        GRPOTrainingGroup(
            group_id=request.rollout_group.group_id,
            profile_id=request.advisor_profile_id,
            candidates=candidates,
            reward_mean=reward_mean,
            reward_std=reward_std,
        )
    ]


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
        training_groups = build_grpo_training_groups(request)
        reward_values = [float(value) for value in request.rollout_group.reward_values]
        rollout_count = request.rollout_group.rollout_count
        mean_reward = round(sum(reward_values) / rollout_count, 4) if reward_values else 0.0
        trainer_result = _normalize_trainer_result(
            self.trainer.train(request, checkpoint_dir, training_groups)
        )
        flattened_candidates = [candidate for group in training_groups for candidate in group.candidates]
        training_metrics = {
            "mean_reward": mean_reward,
            "max_reward": max(reward_values) if reward_values else 0.0,
            "min_reward": min(reward_values) if reward_values else 0.0,
            "rollout_count": rollout_count,
            "num_generations": request.training_config.num_generations,
            "max_steps": request.training_config.max_steps,
            "training_sample_count": len(training_samples),
            "training_group_count": len(training_groups),
            "positive_advantage_count": sum(1 for candidate in flattened_candidates if candidate.advantage > 0),
            "negative_advantage_count": sum(1 for candidate in flattened_candidates if candidate.advantage < 0),
            **trainer_result.metrics,
        }

        checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
        backend_manifest_path = job_dir / "backend-manifest.json"
        artifact_paths = {
            **trainer_result.artifact_paths,
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "backend_manifest": str(backend_manifest_path),
        }
        trajectory_ids = _rollout_trajectory_ids(request)
        lora_target_modules = list(request.training_config.target_modules)

        checkpoint_manifest_path.write_text(
            json.dumps(
                {
                    "checkpoint_id": checkpoint_id,
                    "job_id": request.job_id,
                    "experiment_id": request.experiment_id,
                    "advisor_profile_id": request.advisor_profile_id,
                    "backend_name": self.backend_name,
                    "rollout_group_id": request.rollout_group.group_id,
                    "trajectory_ids": trajectory_ids,
                    "base_model_name": request.training_config.base_model_name,
                    "target_modules": lora_target_modules,
                    "lora_rank": request.training_config.lora_rank,
                    "training_metrics": training_metrics,
                    "rollout_summary": request.rollout_group.summary,
                    "artifact_paths": artifact_paths,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        backend_manifest_path.write_text(
            json.dumps(
                {
                    "job_id": request.job_id,
                    "experiment_id": request.experiment_id,
                    "advisor_profile_id": request.advisor_profile_id,
                    "profile_id": request.advisor_profile_id,
                    "backend_name": self.backend_name,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_path": str(checkpoint_dir),
                    "training_config": request.training_config.model_dump(),
                    "base_model_name": request.training_config.base_model_name,
                    "adapter_method": request.training_config.adapter_method,
                    "lora_rank": request.training_config.lora_rank,
                    "target_modules": lora_target_modules,
                    "rollout_group_id": request.rollout_group.group_id,
                    "trajectory_ids": trajectory_ids,
                    "rollout_summary": request.rollout_group.summary,
                    "training_metrics": training_metrics,
                    "training_sample_count": len(training_samples),
                    "training_group_count": len(training_groups),
                    "artifact_paths": artifact_paths,
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
            artifact_paths=artifact_paths,
            rollout_summary=request.rollout_group.summary,
        )


def _rollout_trajectory_ids(request: TrainingBackendRunRequest) -> list[str]:
    trajectory_ids: list[str] = []
    for result in request.rollout_group.results:
        if not result.trajectory:
            continue
        trajectory_payload = _normalized_payload(result.trajectory)
        trajectory_id = trajectory_payload.get("trajectory_id")
        if trajectory_id and trajectory_id not in trajectory_ids:
            trajectory_ids.append(str(trajectory_id))
    return trajectory_ids


def _normalized_payload(payload: BaseModel | dict) -> dict:
    # Normalize Pydantic models and dicts into a stable JSON shape before sample serialization.
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    return dict(payload)


def _stable_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _trajectory_final_reward(trajectory_payload: dict[str, Any], reward_label: BaseModel | dict) -> float:
    # The rollout reward label is the canonical evaluator output. Persisted trajectory payloads can lag
    # when replayed or repaired, so use trajectory.final_reward only as a compatibility fallback.
    reward_payload = reward_label.model_dump() if isinstance(reward_label, BaseModel) else dict(reward_label)
    if "total_reward" in reward_payload:
        return float(reward_payload["total_reward"])

    final_reward = trajectory_payload.get("final_reward")
    if isinstance(final_reward, BaseModel):
        final_reward = final_reward.model_dump()
    if isinstance(final_reward, dict) and "total_reward" in final_reward:
        return float(final_reward["total_reward"])
    if isinstance(final_reward, int | float):
        return float(final_reward)
    return 0.0


def _sample_signature(sample: GRPOTrainingSample) -> str:
    prompt_payload = _strip_signature_noise(json.loads(sample.prompt))
    completion_payload = _strip_signature_noise(json.loads(sample.completion))
    return _stable_json_dumps(
        {
            "prompt": prompt_payload,
            "completion": completion_payload,
            "reward": sample.reward,
            "profile_id": sample.profile_id,
        }
    )


def _strip_signature_noise(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _strip_signature_noise(value)
            for key, value in sorted(payload.items())
            if key not in {"run_id", "session_id", "trajectory_id"}
        }
    if isinstance(payload, list):
        return [_strip_signature_noise(item) for item in payload]
    return payload


def _reward_total(reward_label: BaseModel | dict) -> float:
    # Reward labels can still arrive as plain dicts in tests and persisted artifact paths.
    if isinstance(reward_label, BaseModel):
        return float(reward_label.model_dump().get("total_reward", 0.0))
    return float(dict(reward_label).get("total_reward", 0.0))


def _population_stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _rounded_reward_stat(value: float) -> float:
    return round(float(value), 6)


def _reward_advantage(reward: float, reward_mean: float, reward_std: float) -> float:
    if reward_std <= 1e-12:
        return _rounded_reward_stat(reward - reward_mean)
    return _rounded_reward_stat((reward - reward_mean) / reward_std)


def _grpo_weighted_loss(model, batch, lengths, advantages):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    token_counts = mx.maximum(mask.sum(axis=1), 1)
    sequence_ce = ce.astype(mx.float32).sum(axis=1) / token_counts.astype(mx.float32)
    weighted_loss = (sequence_ce * advantages.astype(mx.float32)).mean()
    ntoks = mask.sum()

    return weighted_loss, ntoks


def _build_weighted_iterate_batches(advantages: list[float]):
    def iterate_batches(
        dataset,
        batch_size,
        max_seq_length,
        loop=False,
        seed=None,
        comm_group=None,
    ):
        if len(dataset) < batch_size:
            raise ValueError(
                f"Dataset must have at least batch_size={batch_size} examples but only has {len(dataset)}."
            )

        idx = sorted(range(len(dataset)), key=lambda item_index: len(dataset[item_index][0]))

        if comm_group is not None:
            offset = comm_group.rank()
            step = comm_group.size()
        else:
            offset = 0
            step = 1
        if batch_size % step != 0:
            raise ValueError("The batch size must be divisible by the number of workers")

        batch_idx = [
            idx[i + offset : i + offset + batch_size : step]
            for i in range(0, len(idx) - batch_size + 1, batch_size)
        ]
        if seed is not None:
            np.random.seed(seed)

        while True:
            indices = np.random.permutation(len(batch_idx))
            for batch_position in indices:
                selected_idx = batch_idx[int(batch_position)]
                batch = [dataset[item_index] for item_index in selected_idx]
                if len(batch[0]) == 2:
                    batch_tokens, offsets = zip(*batch)
                else:
                    batch_tokens = batch
                    offsets = [0] * len(batch)
                lengths = [len(tokens) for tokens in batch_tokens]
                if max(lengths) > max_seq_length:
                    print(
                        f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                        f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                        "Consider pre-splitting your data to save memory."
                    )

                pad_to = 32
                max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
                max_length_in_batch = min(max_length_in_batch, max_seq_length)
                batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

                for item_offset in range(batch_size // step):
                    truncated_length = min(lengths[item_offset], max_seq_length)
                    batch_arr[item_offset, :truncated_length] = batch_tokens[item_offset][:truncated_length]
                    lengths[item_offset] = truncated_length

                yield (
                    mx.array(batch_arr),
                    mx.array(list(zip(offsets, lengths))),
                    mx.array([advantages[item_index] for item_index in selected_idx], dtype=mx.float32),
                )

            if not loop:
                break

    return iterate_batches


def _normalize_trainer_result(result: TrainerRunArtifact | dict) -> TrainerRunArtifact:
    if isinstance(result, TrainerRunArtifact):
        return result
    return TrainerRunArtifact.model_validate(result)


def _ensure_mlx_training_dependencies() -> None:
    if (
        np is None
        or mx is None
        or nn is None
        or mlx_optim is None
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
