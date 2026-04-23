import json
from pathlib import Path

from agent.advisor import GRPOTrainingBackend, TrainingBackendRunRequest
from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult


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
                "train_loss": 0.125,
                "optimizer_steps": 4,
                "trained_examples": len(training_samples),
            },
        }


def test_grpo_training_backend_writes_checkpoint_and_backend_manifest(tmp_path):
    backend = GRPOTrainingBackend(trainer=StubTrainer())
    training_config = AdvisorTrainingConfig(
        backend="grpo",
        rollout_group_size=2,
        num_generations=4,
        max_steps=6,
        max_prompt_tokens=2048,
        max_completion_tokens=512,
        checkpoint_root="artifacts/checkpoints/coding-default",
        base_model_name="mlx-community/Qwen2.5-3B-Instruct-4bit",
        adapter_method="lora",
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    rollout_group = TrainingRolloutGroupResult(
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
                reward_label={"total_reward": 0.75},
                diagnostics={"multi_turn": False},
            )
        ],
        reward_values=[0.75],
        summary={"mean_reward": 0.75},
    )
    request = TrainingBackendRunRequest(
        job_id="job-1",
        experiment_id="exp-1",
        advisor_profile_id="coding-default",
        training_config=training_config,
        rollout_group=rollout_group,
        output_dir=str(tmp_path / "artifacts"),
    )

    result = backend.run(request)

    checkpoint_path = Path(result.checkpoint_path)
    manifest_path = Path(result.artifact_paths["backend_manifest"])
    checkpoint_manifest_path = Path(result.artifact_paths["checkpoint_manifest"])
    adapter_path = Path(result.artifact_paths["adapter_model"])
    adapter_config_path = Path(result.artifact_paths["adapter_config"])
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    assert result.backend_name == "grpo"
    assert result.advisor_profile_id == "coding-default"
    assert checkpoint_path.exists()
    assert manifest_path.exists()
    assert adapter_path.exists()
    assert adapter_config_path.exists()
    assert result.rollout_summary["mean_reward"] == 0.75
    assert result.training_metrics["mean_reward"] == 0.75
    assert result.training_metrics["train_loss"] == 0.125
    assert result.training_metrics["optimizer_steps"] == 4
    assert checkpoint_manifest["artifact_paths"]["adapter_model"] == str(adapter_path)
    assert checkpoint_manifest["artifact_paths"]["adapter_config"] == str(adapter_config_path)
