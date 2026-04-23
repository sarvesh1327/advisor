from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .reward_model import RewardWeights


def get_default_advisor_home() -> Path:
    # Prefer explicit app state dirs so hosted or multi-profile setups stay relocatable.
    explicit_home = os.getenv("ADVISOR_HOME")
    if explicit_home:
        return Path(explicit_home).expanduser()

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "advisor"

    return Path.home() / ".advisor"


def get_default_profiles_path() -> Path:
    # Keep the default profile registry with the repo config so profile selection stays deterministic in local development.
    return Path(__file__).resolve().parents[2] / "config" / "advisor_profiles.toml"


class AdvisorSettings(BaseModel):
    # Defaults describe the standalone local runtime; callers can override through TOML or env.
    enabled: bool = False
    trace_db_path: str = str(get_default_advisor_home() / "advisor.db")
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    model_version: str = "advisor-qwen25-3b-v1"
    system_prompt: str = (
        "You are an execution advisor. Return ONLY valid JSON with keys "
        "task_type, focus_targets, relevant_files, relevant_symbols, constraints, likely_failure_modes, "
        "recommended_plan, avoid, confidence, notes, injection_policy. "
        "Prefer generic focus_targets as the canonical advice surface. "
        "Do not emit markdown, role tags, or commentary."
    )
    fallback_model_name: str | None = None
    max_context_files: int = 8
    max_tree_entries: int = 60
    max_failures: int = 5
    max_tokens: int = 500
    temperature: float = 0.1
    token_budget: int = 1800
    max_retries: int = 1
    inference_timeout_seconds: int = 20
    warm_load_on_start: bool = False
    enable_fallback_runtime: bool = True
    reward_preset: str = "balanced"
    reward_weights_config: dict[str, float] = Field(default_factory=dict, alias="reward_weights")
    advisor_profile_id: str = "default"
    advisor_profiles_path: str = str(get_default_profiles_path())
    retention_days: int = 30
    event_log_path: str = str(get_default_advisor_home() / "events.jsonl")
    redact_sensitive_fields: bool = True
    hosted_mode: bool = False

    @model_validator(mode="after")
    def validate_ranges(self) -> "AdvisorSettings":
        if self.max_context_files <= 0:
            raise ValueError("max_context_files must be > 0")
        if self.max_tree_entries <= 0:
            raise ValueError("max_tree_entries must be > 0")
        if self.max_failures < 0:
            raise ValueError("max_failures must be >= 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.token_budget < self.max_tokens:
            raise ValueError("token_budget must be >= max_tokens")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be > 0")
        if self.retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        if self.reward_preset not in self.reward_presets():
            raise ValueError(f"reward_preset must be one of {sorted(self.reward_presets())}")
        for name, value in self.reward_weights_config.items():
            if name not in {"task_success", "efficiency", "targeting_quality", "constraint_compliance", "human_usefulness"}:
                raise ValueError(f"unknown reward weight '{name}'")
            if value < 0.0:
                raise ValueError(f"reward weight '{name}' must be >= 0")
        return self

    @classmethod
    def reward_presets(cls) -> dict[str, RewardWeights]:
        return {
            "balanced": RewardWeights(
                task_success=0.35,
                efficiency=0.15,
                targeting_quality=0.2,
                constraint_compliance=0.2,
                human_usefulness=0.1,
            ),
            "conservative": RewardWeights(
                task_success=0.45,
                efficiency=0.1,
                targeting_quality=0.2,
                constraint_compliance=0.2,
                human_usefulness=0.05,
            ),
            "human-first": RewardWeights(
                task_success=0.25,
                efficiency=0.1,
                targeting_quality=0.2,
                constraint_compliance=0.2,
                human_usefulness=0.25,
            ),
        }

    def reward_weights(self) -> RewardWeights:
        preset = self.reward_presets()[self.reward_preset]
        overrides = self.reward_weights_config or {}
        if not overrides:
            return preset
        merged = {
            "task_success": overrides.get("task_success", preset.task_success),
            "efficiency": overrides.get("efficiency", preset.efficiency),
            "targeting_quality": overrides.get("targeting_quality", preset.targeting_quality),
            "constraint_compliance": overrides.get("constraint_compliance", preset.constraint_compliance),
            "human_usefulness": overrides.get("human_usefulness", preset.human_usefulness),
        }
        return RewardWeights(**merged)

    @classmethod
    def from_env(cls) -> "AdvisorSettings":
        advisor_home = get_default_advisor_home()
        default_db = advisor_home / "advisor.db"
        reward_weights = {
            key: float(os.getenv(env_name))
            for key, env_name in {
                "task_success": "ADVISOR_REWARD_TASK_SUCCESS",
                "efficiency": "ADVISOR_REWARD_EFFICIENCY",
                "targeting_quality": "ADVISOR_REWARD_TARGETING_QUALITY",
                "constraint_compliance": "ADVISOR_REWARD_CONSTRAINT_COMPLIANCE",
                "human_usefulness": "ADVISOR_REWARD_HUMAN_USEFULNESS",
            }.items()
            if os.getenv(env_name) is not None
        }
        return cls(
            enabled=os.getenv("ADVISOR_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
            trace_db_path=os.getenv("ADVISOR_TRACE_DB", str(default_db)),
            model_name=os.getenv("ADVISOR_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            model_version=os.getenv("ADVISOR_MODEL_VERSION", "advisor-qwen25-3b-v1"),
            system_prompt=os.getenv(
                "ADVISOR_SYSTEM_PROMPT",
                "You are an execution advisor. Return ONLY valid JSON with keys "
                "task_type, focus_targets, relevant_files, relevant_symbols, constraints, likely_failure_modes, "
                "recommended_plan, avoid, confidence, notes, injection_policy. "
                "Prefer generic focus_targets as the canonical advice surface. "
                "Do not emit markdown, role tags, or commentary.",
            ),
            fallback_model_name=os.getenv("ADVISOR_FALLBACK_MODEL"),
            max_context_files=int(os.getenv("ADVISOR_MAX_CONTEXT_FILES", "8")),
            max_tree_entries=int(os.getenv("ADVISOR_MAX_TREE_ENTRIES", "60")),
            max_failures=int(os.getenv("ADVISOR_MAX_FAILURES", "5")),
            max_tokens=int(os.getenv("ADVISOR_MAX_TOKENS", "500")),
            temperature=float(os.getenv("ADVISOR_TEMPERATURE", "0.1")),
            token_budget=int(os.getenv("ADVISOR_TOKEN_BUDGET", "1800")),
            max_retries=int(os.getenv("ADVISOR_MAX_RETRIES", "1")),
            inference_timeout_seconds=int(os.getenv("ADVISOR_INFERENCE_TIMEOUT_SECONDS", "20")),
            warm_load_on_start=os.getenv("ADVISOR_WARM_LOAD_ON_START", "0").lower() in {"1", "true", "yes", "on"},
            enable_fallback_runtime=os.getenv("ADVISOR_ENABLE_FALLBACK_RUNTIME", "1").lower()
            in {"1", "true", "yes", "on"},
            reward_preset=os.getenv("ADVISOR_REWARD_PRESET", "balanced"),
            reward_weights=reward_weights,
            advisor_profile_id=os.getenv("ADVISOR_PROFILE_ID", "default"),
            advisor_profiles_path=os.getenv("ADVISOR_PROFILES_PATH", str(get_default_profiles_path())),
            retention_days=int(os.getenv("ADVISOR_RETENTION_DAYS", "30")),
            event_log_path=os.getenv("ADVISOR_EVENT_LOG_PATH", str(advisor_home / "events.jsonl")),
            redact_sensitive_fields=os.getenv("ADVISOR_REDACT_SENSITIVE_FIELDS", "1").lower()
            in {"1", "true", "yes", "on"},
            hosted_mode=os.getenv("ADVISOR_HOSTED_MODE", "0").lower() in {"1", "true", "yes", "on"},
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "AdvisorSettings":
        raw = tomllib.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls(**raw)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AdvisorSettings":
        # Explicit config wins; env remains the zero-config fallback.
        explicit_path = config_path or os.getenv("ADVISOR_CONFIG")
        if explicit_path:
            return cls.from_toml(explicit_path)
        return cls.from_env()

    def ensure_dirs(self) -> None:
        Path(self.trace_db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(self.event_log_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    def health_payload(self) -> dict:
        # Health payloads intentionally expose runtime knobs so startup issues are easy to diagnose.
        return {
            "valid": True,
            "enabled": self.enabled,
            "trace_db_path": self.trace_db_path,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "system_prompt": self.system_prompt,
            "fallback_model_name": self.fallback_model_name,
            "max_tokens": self.max_tokens,
            "token_budget": self.token_budget,
            "max_retries": self.max_retries,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "warm_load_on_start": self.warm_load_on_start,
            "enable_fallback_runtime": self.enable_fallback_runtime,
            "reward_preset": self.reward_preset,
            "reward_weights": self.reward_weights().__dict__,
            "advisor_profile_id": self.advisor_profile_id,
            "advisor_profiles_path": self.advisor_profiles_path,
            "retention_days": self.retention_days,
            "event_log_path": self.event_log_path,
            "redact_sensitive_fields": self.redact_sensitive_fields,
            "hosted_mode": self.hosted_mode,
        }
