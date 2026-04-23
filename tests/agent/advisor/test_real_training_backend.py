import json

from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training_backends import (
    GRPOTrainingSample,
    TrainingBackendRunRequest,
    build_grpo_training_samples,
)
from agent.advisor.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult


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
