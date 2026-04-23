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

            [profiles.image-ui]
            domain = "image-ui"
            description = "UI/image advisor profile"
            """
        ).strip()
    )

    registry = AdvisorProfileRegistry.from_toml(config_path)

    assert registry.default_profile_id == "coding-default"
    assert registry.get("coding-default").domain == "coding"
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
            """
        ).strip()
    )

    registry = AdvisorProfileRegistry.from_toml(config_path)
    profile = registry.resolve("coding-default")

    assert profile.training is not None
    assert profile.training.backend == "grpo"
    assert profile.training.rollout_group_size == 4
    assert profile.training.num_generations == 8
    assert profile.training.checkpoint_root == "artifacts/checkpoints/coding-default"
