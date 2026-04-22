import os
from pathlib import Path

from agent.advisor.settings import AdvisorSettings, get_default_advisor_home



def test_get_default_advisor_home_prefers_explicit_env(monkeypatch, tmp_path):
    custom_home = tmp_path / "advisor-home"
    monkeypatch.setenv("ADVISOR_HOME", str(custom_home))
    assert get_default_advisor_home() == custom_home



def test_settings_from_env_uses_advisor_prefix(monkeypatch, tmp_path):
    advisor_home = tmp_path / "product-home"
    monkeypatch.setenv("ADVISOR_HOME", str(advisor_home))
    monkeypatch.setenv("ADVISOR_ENABLED", "true")
    monkeypatch.setenv("ADVISOR_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    monkeypatch.setenv("ADVISOR_MODEL_VERSION", "advisor-qwen25-7b-v1")
    monkeypatch.setenv("ADVISOR_MAX_TOKENS", "700")
    monkeypatch.setenv("ADVISOR_TEMPERATURE", "0.2")
    monkeypatch.setenv("ADVISOR_TOKEN_BUDGET", "2200")

    settings = AdvisorSettings.from_env()

    assert settings.enabled is True
    assert settings.trace_db_path == str(advisor_home / "advisor.db")
    assert settings.model_name == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert settings.model_version == "advisor-qwen25-7b-v1"
    assert settings.max_tokens == 700
    assert settings.temperature == 0.2
    assert settings.token_budget == 2200



def test_settings_ensure_dirs_creates_parent(tmp_path):
    db_path = tmp_path / "nested" / "data" / "advisor.db"
    settings = AdvisorSettings(trace_db_path=str(db_path))
    settings.ensure_dirs()
    assert db_path.parent.exists()
