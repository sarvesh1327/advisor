import json
from pathlib import Path

from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training.training_backends import (
    GRPOTrainingBackend,
    GRPOTrainingSample,
    TrainingBackendRunRequest,
    build_grpo_training_groups,
    build_grpo_training_samples,
)
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult


def _training_request() -> TrainingBackendRunRequest:
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
                packet={
                    "run_id": "run-1",
                    "task_text": "repair main.py",
                    "task_type": "bugfix",
                    "repo": {"path": "/tmp/repo", "branch": "main"},
                },
                primary_advice={
                    "task_type": "bugfix",
                    "recommended_plan": ["inspect main.py", "run pytest -q"],
                    "confidence": 0.8,
                },
                executor_result={"status": "success", "output": "patched main.py"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.75},
                diagnostics={"multi_turn": False},
            ),
            TrainingRolloutResult(
                rollout_id="rollout-2",
                advisor_profile_id="coding-default",
                packet={
                    "run_id": "run-2",
                    "task_text": "repair utils.py",
                    "task_type": "bugfix",
                    "repo": {"path": "/tmp/repo", "branch": "main"},
                },
                primary_advice={
                    "task_type": "bugfix",
                    "recommended_plan": ["inspect utils.py", "run pytest tests/test_utils.py -q"],
                    "confidence": 0.7,
                },
                executor_result={"status": "success", "output": "patched utils.py"},
                verifier_results=[],
                outcome={"status": "success"},
                reward_label={"total_reward": 0.5},
                diagnostics={"multi_turn": False},
            ),
        ],
        reward_values=[0.75, 0.5],
        summary={"mean_reward": 0.625},
    )
    return TrainingBackendRunRequest(
        job_id="job-1",
        experiment_id="exp-1",
        advisor_profile_id="coding-default",
        training_config=training_config,
        rollout_group=rollout_group,
        output_dir="/tmp/artifacts",
    )


class RecordingTrainer:
    def __init__(self):
        self.calls = []

    def train(self, request, checkpoint_dir, training_groups):
        flattened_candidates = [candidate for group in training_groups for candidate in group.candidates]
        self.calls.append(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "group_count": len(training_groups),
                "candidate_count": len(flattened_candidates),
                "rewards": [candidate.reward for candidate in flattened_candidates],
                "advantages": [candidate.advantage for candidate in flattened_candidates],
                "group_ids": [group.group_id for group in training_groups],
            }
        )
        adapter_path = checkpoint_dir / "adapters.safetensors"
        adapter_path.write_bytes(b"trained-adapter")
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
                "train_loss": 0.2,
                "optimizer_steps": 8,
                "trained_examples": len(flattened_candidates),
            },
        }



def test_build_grpo_training_groups_preserves_grouped_candidates_and_advantages():
    request = _training_request()

    groups = build_grpo_training_groups(request)

    assert len(groups) == 1
    group = groups[0]
    assert group.group_id == "group-1"
    assert group.profile_id == "coding-default"
    assert group.reward_mean == 0.625
    assert group.reward_std == 0.125
    assert [candidate.reward for candidate in group.candidates] == [0.75, 0.5]
    assert [candidate.advantage for candidate in group.candidates] == [1.0, -1.0]
    assert [candidate.group_id for candidate in group.candidates] == ["group-1", "group-1"]
    assert [candidate.candidate_index for candidate in group.candidates] == [0, 1]
    assert [candidate.provenance["rollout_id"] for candidate in group.candidates] == ["rollout-1", "rollout-2"]

    first_prompt = json.loads(group.candidates[0].prompt)
    first_completion = json.loads(group.candidates[0].completion)
    second_prompt = json.loads(group.candidates[1].prompt)
    second_completion = json.loads(group.candidates[1].completion)

    assert first_prompt["task_text"] == "repair main.py"
    assert second_prompt["task_text"] == "repair utils.py"
    assert first_completion["recommended_plan"] == ["inspect main.py", "run pytest -q"]
    assert second_completion["recommended_plan"] == ["inspect utils.py", "run pytest tests/test_utils.py -q"]



def test_build_grpo_training_groups_changes_advantages_when_only_rewards_change():
    request = _training_request()
    low_reward_result = request.rollout_group.results[0].model_copy(
        update={"reward_label": {"total_reward": 0.1}}
    )
    high_reward_result = request.rollout_group.results[1].model_copy(
        update={"reward_label": {"total_reward": 0.9}}
    )
    mutated_request = request.model_copy(
        update={
            "rollout_group": request.rollout_group.model_copy(
                update={
                    "results": [low_reward_result, high_reward_result],
                    "reward_values": [0.1, 0.9],
                    "summary": {"mean_reward": 0.5},
                }
            )
        }
    )

    baseline_groups = build_grpo_training_groups(request)
    mutated_groups = build_grpo_training_groups(mutated_request)

    assert [candidate.reward for candidate in baseline_groups[0].candidates] == [0.75, 0.5]
    assert [candidate.reward for candidate in mutated_groups[0].candidates] == [0.1, 0.9]
    assert [candidate.advantage for candidate in baseline_groups[0].candidates] == [1.0, -1.0]
    assert [candidate.advantage for candidate in mutated_groups[0].candidates] == [-1.0, 1.0]



def test_build_grpo_training_samples_is_deterministic():
    request = _training_request()

    first = [sample.model_dump() for sample in build_grpo_training_samples(request)]
    second = [sample.model_dump() for sample in build_grpo_training_samples(request)]

    assert first == second



def test_build_grpo_training_samples_preserves_rewards_and_provenance():
    request = _training_request()

    samples = build_grpo_training_samples(request)

    assert [sample.reward for sample in samples] == [0.75, 0.5]
    assert [sample.provenance["rollout_id"] for sample in samples] == ["rollout-1", "rollout-2"]
    assert all(sample.profile_id == "coding-default" for sample in samples)
    assert all(sample.provenance["job_id"] == "job-1" for sample in samples)



def test_build_grpo_training_samples_serialize_prompt_and_completion_for_advisor_training():
    request = _training_request()

    sample = build_grpo_training_samples(request)[0]

    assert isinstance(sample, GRPOTrainingSample)
    prompt_payload = json.loads(sample.prompt)
    completion_payload = json.loads(sample.completion)
    assert prompt_payload["task_text"] == "repair main.py"
    assert completion_payload["recommended_plan"] == ["inspect main.py", "run pytest -q"]



def test_grpo_training_backend_records_trainer_metrics_and_adapter_artifacts(tmp_path):
    trainer = RecordingTrainer()
    request = _training_request().model_copy(update={"output_dir": str(tmp_path / "artifacts")})

    result = GRPOTrainingBackend(trainer=trainer).run(request)

    adapter_path = Path(result.artifact_paths["adapter_model"])
    config_path = Path(result.artifact_paths["adapter_config"])
    checkpoint_manifest = json.loads(Path(result.artifact_paths["checkpoint_manifest"]).read_text(encoding="utf-8"))

    assert trainer.calls[0]["group_count"] == 1
    assert trainer.calls[0]["candidate_count"] == 2
    assert trainer.calls[0]["group_ids"] == ["group-1"]
    assert trainer.calls[0]["rewards"] == [0.75, 0.5]
    assert trainer.calls[0]["advantages"] == [1.0, -1.0]
    assert adapter_path.exists()
    assert config_path.exists()
    assert result.training_metrics["train_loss"] == 0.2
    assert result.training_metrics["optimizer_steps"] == 8
    assert checkpoint_manifest["artifact_paths"]["adapter_model"] == str(adapter_path)
    assert checkpoint_manifest["artifact_paths"]["adapter_config"] == str(config_path)



def test_grpo_training_backend_persists_lora_config_in_adapter_manifest(tmp_path):
    trainer = RecordingTrainer()
    request = _training_request().model_copy(update={"output_dir": str(tmp_path / "artifacts")})

    result = GRPOTrainingBackend(trainer=trainer).run(request)

    adapter_config = json.loads(Path(result.artifact_paths["adapter_config"]).read_text(encoding="utf-8"))

    assert adapter_config["adapter_method"] == "lora"
    assert adapter_config["lora_rank"] == 32
    assert adapter_config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
