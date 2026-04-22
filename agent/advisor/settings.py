from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, model_validator


def get_default_advisor_home() -> Path:
    explicit_home = os.getenv("ADVISOR_HOME")
    if explicit_home:
        return Path(explicit_home).expanduser()

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "advisor"

    return Path.home() / ".advisor"


class AdvisorSettings(BaseModel):
    enabled: bool = False
    trace_db_path: str = str(get_default_advisor_home() / "advisor.db")
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    model_version: str = "advisor-qwen25-3b-v1"
    system_prompt: str = (
        "You are an execution advisor. Return ONLY valid JSON with keys "
        "task_type, relevant_files, relevant_symbols, constraints, likely_failure_modes, "
        "recommended_plan, avoid, confidence, notes. Do not emit markdown, role tags, or commentary."
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
        return self

    @classmethod
    def from_env(cls) -> "AdvisorSettings":
        advisor_home = get_default_advisor_home()
        default_db = advisor_home / "advisor.db"
        return cls(
            enabled=os.getenv("ADVISOR_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
            trace_db_path=os.getenv("ADVISOR_TRACE_DB", str(default_db)),
            model_name=os.getenv("ADVISOR_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            model_version=os.getenv("ADVISOR_MODEL_VERSION", "advisor-qwen25-3b-v1"),
            system_prompt=os.getenv(
                "ADVISOR_SYSTEM_PROMPT",
                "You are an execution advisor. Return ONLY valid JSON with keys "
                "task_type, relevant_files, relevant_symbols, constraints, likely_failure_modes, "
                "recommended_plan, avoid, confidence, notes. Do not emit markdown, role tags, or commentary.",
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
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "AdvisorSettings":
        raw = tomllib.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls(**raw)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AdvisorSettings":
        explicit_path = config_path or os.getenv("ADVISOR_CONFIG")
        if explicit_path:
            return cls.from_toml(explicit_path)
        return cls.from_env()

    def ensure_dirs(self) -> None:
        Path(self.trace_db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    def health_payload(self) -> dict:
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
        }
