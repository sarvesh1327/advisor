from pathlib import Path

from agent.advisor import GRPOTrainingBackend, TrainingBackendRunRequest
from agent.advisor.profiles import AdvisorTrainingConfig
from agent.advisor.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult


def test_grpo_training_backend_writes_checkpoint_and_backend_manifest(tmp_path):
    backend = GRPOTrainingBackend()
    training_config = AdvisorTrainingConfig(
        backend="grpo",
        rollout_group_size=2,
        num_generations=4,
        max_steps=6,
        max_prompt_tokens=2048,
        max_completion_tokens=512,
        checkpoint_root="artifacts/checkpoints/coding-default",
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
    assert result.backend_name == "grpo"
    assert result.advisor_profile_id == "coding-default"
    assert checkpoint_path.exists()
    assert manifest_path.exists()
    assert result.rollout_summary["mean_reward"] == 0.75
    assert result.training_metrics["mean_reward"] == 0.75
