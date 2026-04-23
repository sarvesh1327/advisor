import textwrap

import pytest

from agent.advisor.reward_model import RewardWeights
from agent.advisor.settings import (
    AdvisorSettings,
    get_default_advisor_home,
    get_default_profiles_path,
)


def test_get_default_advisor_home_prefers_explicit_env(monkeypatch, tmp_path):
    custom_home = tmp_path / "advisor-home"
    monkeypatch.setenv("ADVISOR_HOME", str(custom_home))
    assert get_default_advisor_home() == custom_home



def test_get_default_profiles_path_points_to_repo_config():
    assert get_default_profiles_path().name == "advisor_profiles.toml"
    assert get_default_profiles_path().parent.name == "config"



def test_settings_from_env_uses_advisor_prefix(monkeypatch, tmp_path):
    advisor_home = tmp_path / "product-home"
    monkeypatch.setenv("ADVISOR_HOME", str(advisor_home))
    monkeypatch.setenv("ADVISOR_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    monkeypatch.setenv("ADVISOR_MODEL_VERSION", "advisor-qwen25-7b-v1")
    monkeypatch.setenv("ADVISOR_SYSTEM_PROMPT", "You are a generic execution advisor.")
    monkeypatch.setenv("ADVISOR_MAX_TOKENS", "700")
    monkeypatch.setenv("ADVISOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("ADVISOR_TOKEN_BUDGET", "2200")
    monkeypatch.setenv("ADVISOR_MAX_RETRIES", "3")
    monkeypatch.setenv("ADVISOR_INFERENCE_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("ADVISOR_REWARD_PRESET", "human-first")
    monkeypatch.setenv("ADVISOR_RETENTION_DAYS", "45")
    monkeypatch.setenv("ADVISOR_EVENT_LOG_PATH", str(advisor_home / "logs" / "events.jsonl"))
    monkeypatch.setenv("ADVISOR_REDACT_SENSITIVE_FIELDS", "true")
    monkeypatch.setenv("ADVISOR_HOSTED_MODE", "true")
    monkeypatch.setenv("ADVISOR_PROFILE_ID", "coding-default")
    monkeypatch.setenv("ADVISOR_PROFILES_PATH", str(tmp_path / "profiles.toml"))

    settings = AdvisorSettings.from_env()

    assert settings.enabled is True
    assert settings.trace_db_path == str(advisor_home / "advisor.db")
    assert settings.model_name == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert settings.model_version == "advisor-qwen25-7b-v1"
    assert settings.system_prompt == "You are a generic execution advisor."
    assert settings.max_tokens == 700
    assert settings.temperature == 0.2
    assert settings.token_budget == 2200
    assert settings.max_retries == 3
    assert settings.inference_timeout_seconds == 15
    assert settings.reward_preset == "human-first"
    assert settings.reward_weights().human_usefulness == 0.25
    assert settings.retention_days == 45
    assert settings.event_log_path == str(advisor_home / "logs" / "events.jsonl")
    assert settings.redact_sensitive_fields is True
    assert settings.hosted_mode is True
    assert settings.advisor_profile_id == "coding-default"
    assert settings.advisor_profiles_path == str(tmp_path / "profiles.toml")



def test_settings_from_toml_file_loads_values(tmp_path):
    config_path = tmp_path / "advisor.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            enabled = true
            trace_db_path = "/tmp/advisor/custom.db"
            model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
            model_version = "advisor-qwen25-7b-v2"
            system_prompt = "You are a generic execution advisor."
            max_context_files = 10
            max_tree_entries = 80
            max_failures = 7
            max_tokens = 640
            temperature = 0.3
            token_budget = 2400
            max_retries = 2
            inference_timeout_seconds = 12
            warm_load_on_start = true
            enable_fallback_runtime = true
            fallback_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
            reward_preset = "conservative"
            advisor_profile_id = "image-ui"
            advisor_profiles_path = "/tmp/advisor/profiles.toml"
            [reward_weights]
            task_success = 0.4
            efficiency = 0.1
            targeting_quality = 0.2
            constraint_compliance = 0.2
            human_usefulness = 0.1
            """
        ).strip()
    )

    settings = AdvisorSettings.from_toml(config_path)

    assert settings.enabled is True
    assert settings.trace_db_path == "/tmp/advisor/custom.db"
    assert settings.model_version == "advisor-qwen25-7b-v2"
    assert settings.system_prompt == "You are a generic execution advisor."
    assert settings.max_failures == 7
    assert settings.token_budget == 2400
    assert settings.max_retries == 2
    assert settings.inference_timeout_seconds == 12
    assert settings.warm_load_on_start is True
    assert settings.enable_fallback_runtime is True
    assert settings.fallback_model_name == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert settings.reward_preset == "conservative"
    assert settings.advisor_profile_id == "image-ui"
    assert settings.advisor_profiles_path == "/tmp/advisor/profiles.toml"
    assert settings.reward_weights() == RewardWeights(
        task_success=0.4,
        efficiency=0.1,
        targeting_quality=0.2,
        constraint_compliance=0.2,
        human_usefulness=0.1,
    )



def test_settings_load_prefers_explicit_config_path(monkeypatch, tmp_path):
    config_path = tmp_path / "advisor.toml"
    config_path.write_text('model_version = "advisor-from-file"\n')
    monkeypatch.setenv("ADVISOR_CONFIG", str(config_path))
    monkeypatch.setenv("ADVISOR_MODEL_VERSION", "advisor-from-env")

    settings = AdvisorSettings.load()

    assert settings.model_version == "advisor-from-file"



def test_settings_reward_weights_support_named_presets_and_validation():
    assert AdvisorSettings(reward_preset="balanced").reward_weights() == RewardWeights(
        task_success=0.35,
        efficiency=0.15,
        targeting_quality=0.2,
        constraint_compliance=0.2,
        human_usefulness=0.1,
    )
    assert AdvisorSettings(reward_preset="human-first").reward_weights().human_usefulness == 0.25
    with pytest.raises(ValueError):
        AdvisorSettings(reward_preset="unknown")
    with pytest.raises(ValueError):
        AdvisorSettings(max_tokens=900, token_budget=800)



def test_settings_validate_rejects_negative_timeout():
    with pytest.raises(ValueError):
        AdvisorSettings(inference_timeout_seconds=0)



def test_settings_ensure_dirs_creates_parent(tmp_path):
    db_path = tmp_path / "nested" / "data" / "advisor.db"
    event_log_path = tmp_path / "logs" / "events.jsonl"
    settings = AdvisorSettings(trace_db_path=str(db_path), event_log_path=str(event_log_path))
    settings.ensure_dirs()
    assert db_path.parent.exists()
    assert event_log_path.parent.exists()
