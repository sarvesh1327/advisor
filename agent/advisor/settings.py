from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


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
    max_context_files: int = 8
    max_tree_entries: int = 60
    max_failures: int = 5
    max_tokens: int = 500
    temperature: float = 0.1
    token_budget: int = 1800

    @classmethod
    def from_env(cls) -> "AdvisorSettings":
        advisor_home = get_default_advisor_home()
        default_db = advisor_home / "advisor.db"
        return cls(
            enabled=os.getenv("ADVISOR_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
            trace_db_path=os.getenv("ADVISOR_TRACE_DB", str(default_db)),
            model_name=os.getenv("ADVISOR_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            model_version=os.getenv("ADVISOR_MODEL_VERSION", "advisor-qwen25-3b-v1"),
            max_tokens=int(os.getenv("ADVISOR_MAX_TOKENS", "500")),
            temperature=float(os.getenv("ADVISOR_TEMPERATURE", "0.1")),
            token_budget=int(os.getenv("ADVISOR_TOKEN_BUDGET", "1800")),
        )

    def ensure_dirs(self) -> None:
        Path(self.trace_db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
