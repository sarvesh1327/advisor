import textwrap

import pytest

from agent.advisor.profiles import AdvisorProfileRegistry


def test_profile_registry_loads_profiles_from_toml(tmp_path):
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            default_profile_id = "coding-default"

            [profiles.coding-default]
            domain = "coding"
            description = "Default coding advisor profile"

            [profiles.researcher]
            domain = "research-writing"
            description = "Research and writing advisor profile"

            [profiles.image-ui]
            domain = "image-ui"
            description = "UI/image advisor profile"
            """
        ).strip()
    )

    registry = AdvisorProfileRegistry.from_toml(config_path)

    assert registry.default_profile_id == "coding-default"
    assert registry.get("coding-default").domain == "coding"
    assert registry.get("researcher").domain == "research-writing"
    assert registry.get("image-ui").description == "UI/image advisor profile"



def test_profile_registry_rejects_unknown_default_profile(tmp_path):
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            default_profile_id = "missing-profile"

            [profiles.coding-default]
            domain = "coding"
            """
        ).strip()
    )

    with pytest.raises(ValueError, match="default_profile_id"):
        AdvisorProfileRegistry.from_toml(config_path)


def test_profile_registry_loads_profile_training_config_from_toml(tmp_path):
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            default_profile_id = "coding-default"

            [profiles.coding-default]
            domain = "coding"
            description = "Default coding advisor profile"

            [profiles.coding-default.training]
            backend = "grpo"
            rollout_group_size = 4
            num_generations = 8
            max_steps = 12
            max_prompt_tokens = 4096
            max_completion_tokens = 1024
            checkpoint_root = "artifacts/checkpoints/coding-default"
            base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
            adapter_method = "lora"
            lora_rank = 32
            lora_alpha = 64
            lora_dropout = 0.05
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            [profiles.researcher]
            domain = "research-writing"
            reward_spec_id = "research_writing_match"
            description = "Research and writing advisor profile"

            [profiles.researcher.training]
            backend = "grpo"
            rollout_group_size = 4
            num_generations = 8
            max_steps = 12
            max_prompt_tokens = 4096
            max_completion_tokens = 1024
            checkpoint_root = "artifacts/checkpoints/researcher"
            base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
            adapter_method = "lora"
            lora_rank = 32
            lora_alpha = 64
            lora_dropout = 0.05
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            """
        ).strip()
    )

    registry = AdvisorProfileRegistry.from_toml(config_path)
    coding_profile = registry.resolve("coding-default")
    researcher_profile = registry.resolve("researcher")

    assert coding_profile.training is not None
    assert coding_profile.training.backend == "grpo"
    assert coding_profile.training.rollout_group_size == 4
    assert coding_profile.training.num_generations == 8
    assert coding_profile.training.checkpoint_root == "artifacts/checkpoints/coding-default"
    assert coding_profile.training.base_model_name == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert coding_profile.training.adapter_method == "lora"
    assert coding_profile.training.lora_rank == 32
    assert coding_profile.training.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert researcher_profile.reward_spec_id == "research_writing_match"
    assert researcher_profile.training is not None
    assert researcher_profile.training.checkpoint_root == "artifacts/checkpoints/researcher"



def test_live_profile_registry_covers_full_phase2_matrix():
    registry = AdvisorProfileRegistry.from_toml("config/advisor_profiles.toml")

    assert registry.default_profile_id == "generalist"
    assert set(registry.profiles) == {"coding-default", "researcher", "text-ui", "image-ui", "generalist"}
    assert registry.get("coding-default").domain == "coding"
    assert registry.get("researcher").domain == "research-writing"
    assert registry.get("text-ui").domain == "text-ui"
    assert registry.get("image-ui").domain == "image-ui"
    assert registry.get("generalist").domain == "conversation"
    assert registry.get("text-ui").training is not None
    assert registry.get("image-ui").training is not None
    assert registry.get("generalist").reward_spec_id == "generalist_multi_turn_conversation"
    assert registry.get("generalist").training is not None
    assert registry.get("generalist").training.checkpoint_root == "checkpoints/generalist"
