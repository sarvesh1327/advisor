import builtins
import importlib.util
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


def _multiturn_training_request() -> TrainingBackendRunRequest:
    request = _training_request()
    trajectory = {
        "trajectory_id": "traj-1",
        "run_id": "run-traj-1",
        "advisor_profile_id": "coding-default",
        "task_text": "repair flaky workflow",
        "turns": [
            {
                "turn_index": 0,
                "state_packet": {
                    "run_id": "volatile-run-0",
                    "task_text": "repair flaky workflow",
                    "task_type": "bugfix",
                    "context": {"metadata": {"session_id": "volatile-session-0"}},
                },
                "advice": {"recommended_plan": ["inspect failing workflow"], "confidence": 0.6},
                "observation": {
                    "turn_index": 0,
                    "status": "partial",
                    "summary": "workflow still fails",
                    "executor_output": "pytest failed",
                    "metrics": {"run_id": "volatile-step-0", "attempt": 1},
                },
            },
            {
                "turn_index": 1,
                "state_packet": {
                    "run_id": "volatile-run-1",
                    "task_text": "repair flaky workflow",
                    "task_type": "bugfix",
                    "history": [{"kind": "turn-observation", "summary": "workflow still fails"}],
                },
                "advice": {"recommended_plan": ["patch retry cleanup"], "confidence": 0.8},
                "observation": {
                    "turn_index": 1,
                    "status": "success",
                    "summary": "workflow passes",
                    "executor_output": "pytest passed",
                    "metrics": {"run_id": "volatile-step-1", "attempt": 2},
                },
            },
        ],
        "final_reward": {"total_reward": 0.72},
        "stop_reason": "success",
        "budget": {"max_turns": 2},
    }
    result = request.rollout_group.results[0].model_copy(
        update={
            "rollout_id": "rollout-traj-1",
            "packet": {"run_id": "legacy-run", "task_text": "legacy packet"},
            "primary_advice": {"recommended_plan": ["legacy advice"]},
            "reward_label": {"total_reward": 0.72},
            "diagnostics": {"multi_turn": True, "turn_count": 2},
            "trajectory": trajectory,
        }
    )
    return request.model_copy(
        update={
            "rollout_group": request.rollout_group.model_copy(
                update={"results": [result], "reward_values": [0.72], "summary": {"mean_reward": 0.72}}
            )
        }
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



def test_training_backends_module_imports_without_numpy_when_training_stack_is_missing(monkeypatch):
    module_path = Path("agent/advisor/training/training_backends.py").resolve()
    module_name = "training_backends_without_numpy_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" or name.startswith("mlx") or name.startswith("mlx_lm"):
            raise ImportError(f"blocked optional dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec.loader.exec_module(module)

    assert module.np is None
    assert module.mx is None
    assert module.nn is None
    assert hasattr(module, "GRPOTrainingBackend")
    assert hasattr(module, "TrainingBackendRunRequest")



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



def test_build_grpo_training_samples_emits_one_sample_per_trajectory_turn_with_provenance():
    request = _multiturn_training_request()

    samples = build_grpo_training_samples(request)

    assert len(samples) == 2
    assert [sample.reward for sample in samples] == [0.72, 0.72]
    assert [sample.profile_id for sample in samples] == ["coding-default", "coding-default"]
    assert [sample.provenance["trajectory_id"] for sample in samples] == ["traj-1", "traj-1"]
    assert [sample.provenance["turn_index"] for sample in samples] == [0, 1]
    assert [sample.provenance["final_reward"] for sample in samples] == [0.72, 0.72]
    assert [sample.provenance["profile_id"] for sample in samples] == ["coding-default", "coding-default"]

    first_prompt = json.loads(samples[0].prompt)
    first_completion = json.loads(samples[0].completion)
    assert first_prompt["state_t"]["task_text"] == "repair flaky workflow"
    assert first_prompt["observation_t"]["summary"] == "workflow still fails"
    assert first_prompt["trajectory_id"] == "traj-1"
    assert first_prompt["turn_index"] == 0
    assert first_prompt["final_reward"] == 0.72
    assert first_prompt["profile_id"] == "coding-default"
    assert first_completion["advice_t"]["recommended_plan"] == ["inspect failing workflow"]



def test_build_grpo_training_samples_uses_reward_label_when_trajectory_reward_is_stale():
    request = _multiturn_training_request()
    stale_trajectory = json.loads(json.dumps(request.rollout_group.results[0].trajectory))
    stale_trajectory["final_reward"] = {"total_reward": -0.25}
    stale_result = request.rollout_group.results[0].model_copy(update={"trajectory": stale_trajectory})
    mutated_request = request.model_copy(
        update={
            "rollout_group": request.rollout_group.model_copy(
                update={"results": [stale_result], "reward_values": [0.72], "summary": {"mean_reward": 0.72}}
            )
        }
    )

    samples = build_grpo_training_samples(mutated_request)
    prompt_payloads = [json.loads(sample.prompt) for sample in samples]

    assert [sample.reward for sample in samples] == [0.72, 0.72]
    assert [payload["final_reward"] for payload in prompt_payloads] == [0.72, 0.72]
    assert [sample.provenance["final_reward"] for sample in samples] == [0.72, 0.72]



def test_build_grpo_training_samples_deduplicates_volatile_ids_but_keeps_distinct_observations():
    request = _multiturn_training_request()
    base_trajectory = request.rollout_group.results[0].trajectory
    duplicate_trajectory = json.loads(json.dumps(base_trajectory))
    duplicate_trajectory["trajectory_id"] = "traj-duplicate-volatile"
    duplicate_trajectory["run_id"] = "different-volatile-run"
    duplicate_trajectory["turns"][0]["state_packet"]["run_id"] = "different-volatile-state-run"
    duplicate_trajectory["turns"][0]["state_packet"]["context"]["metadata"]["session_id"] = "different-session"
    duplicate_trajectory["turns"][0]["observation"]["metrics"]["run_id"] = "different-step-run"
    duplicate_trajectory["turns"] = [duplicate_trajectory["turns"][0]]

    distinct_trajectory = json.loads(json.dumps(base_trajectory))
    distinct_trajectory["trajectory_id"] = "traj-distinct-observation"
    distinct_trajectory["turns"] = [distinct_trajectory["turns"][0]]
    distinct_trajectory["turns"][0]["observation"]["summary"] = "workflow fails with timeout"

    duplicate_result = request.rollout_group.results[0].model_copy(
        update={"rollout_id": "rollout-volatile-duplicate", "trajectory": duplicate_trajectory}
    )
    distinct_result = request.rollout_group.results[0].model_copy(
        update={"rollout_id": "rollout-distinct-observation", "trajectory": distinct_trajectory}
    )
    mutated_request = request.model_copy(
        update={
            "rollout_group": request.rollout_group.model_copy(
                update={
                    "results": [request.rollout_group.results[0], duplicate_result, distinct_result],
                    "reward_values": [0.72, 0.72, 0.72],
                }
            )
        }
    )

    samples = build_grpo_training_samples(mutated_request)
    prompt_payloads = [json.loads(sample.prompt) for sample in samples]

    assert len(samples) == 3
    assert [payload["observation_t"]["summary"] for payload in prompt_payloads] == [
        "workflow still fails",
        "workflow passes",
        "workflow fails with timeout",
    ]



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
