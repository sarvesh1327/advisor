from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class AutonomousLearningPolicy(BaseModel):
    min_fresh_runs: int = 2
    min_distinct_rewards: int = 2
    min_seconds_between_cycles: int = 0
    backoff_seconds: int = 300
    max_consecutive_failures: int = 3
    tick_interval_seconds: int = 60
    max_profiles_per_tick: int = 1
    required_profiles: list[str] = Field(default_factory=list)


class ProfileLearningState(BaseModel):
    advisor_profile_id: str
    consumed_run_ids: list[str] = Field(default_factory=list)
    last_cycle_started_at: str | None = None
    last_cycle_completed_at: str | None = None
    last_cycle_experiment_id: str | None = None
    active_cycle_job_ids: list[str] = Field(default_factory=list)
    consecutive_failures: int = 0
    backoff_until: str | None = None
    paused: bool = False
    paused_reason: str | None = None
    latest_readiness_summary: dict = Field(default_factory=dict)


class AutonomousLearningState(BaseModel):
    controller_paused: bool = False
    controller_paused_reason: str | None = None
    last_tick_at: str | None = None
    latest_validation_summary: dict = Field(default_factory=dict)
    policy: AutonomousLearningPolicy = Field(default_factory=AutonomousLearningPolicy)
    profiles: dict[str, ProfileLearningState] = Field(default_factory=dict)


class AutonomousLearningStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AutonomousLearningState:
        if not self.path.exists():
            return AutonomousLearningState()
        return AutonomousLearningState.model_validate(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, state: AutonomousLearningState) -> AutonomousLearningState:
        self.path.write_text(json.dumps(state.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return state


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def state_path_for_root(root: str | Path) -> Path:
    return Path(root).expanduser() / "learning" / "controller-state.json"
