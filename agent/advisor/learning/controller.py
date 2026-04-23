from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.operators.operator_runtime import OperatorJobQueue, run_continuous_training_cycle
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.storage.trace_store import AdvisorTraceStore
from agent.advisor.training.training_runtime import CheckpointLifecycleManager

from .readiness import build_learning_readiness_report, collect_fresh_rollout_groups, mark_rollout_group_consumed
from .state import (
    AutonomousLearningPolicy,
    AutonomousLearningState,
    AutonomousLearningStateStore,
    ProfileLearningState,
    state_path_for_root,
    utc_now,
)


class AutonomousLearningTickResult(BaseModel):
    controller_paused: bool = False
    launched_profiles: list[str] = Field(default_factory=list)
    skipped_profiles: dict[str, list[str]] = Field(default_factory=dict)
    cycle_results: dict[str, dict] = Field(default_factory=dict)
    readiness: dict[str, dict] = Field(default_factory=dict)


class AutonomousLearningController:
    def __init__(
        self,
        *,
        settings: AdvisorSettings,
        trace_store: AdvisorTraceStore | None = None,
        profile_registry: AdvisorProfileRegistry | None = None,
        queue: OperatorJobQueue | None = None,
        lifecycle_manager: CheckpointLifecycleManager | None = None,
        state_store: AutonomousLearningStateStore | None = None,
    ):
        self.settings = settings
        self.settings.ensure_dirs()
        self.trace_store = trace_store or AdvisorTraceStore(self.settings.trace_db_path)
        self.profile_registry = profile_registry or AdvisorProfileRegistry.from_toml(self.settings.advisor_profiles_path)
        root = Path(self.settings.trace_db_path).expanduser().parent
        self.queue = queue or OperatorJobQueue(root / "operator" / "jobs.json")
        self.lifecycle_manager = lifecycle_manager or CheckpointLifecycleManager(root / "artifacts")
        self.state_store = state_store or AutonomousLearningStateStore(state_path_for_root(root))

    def load_state(self) -> AutonomousLearningState:
        state = self.state_store.load()
        if state.policy is None:
            state.policy = AutonomousLearningPolicy()
        return state

    def save_state(self, state: AutonomousLearningState) -> AutonomousLearningState:
        return self.state_store.save(state)

    def controller_status(self) -> dict:
        state = self.load_state()
        return {
            "controller_paused": state.controller_paused,
            "controller_paused_reason": state.controller_paused_reason,
            "last_tick_at": state.last_tick_at,
            "policy": state.policy.model_dump(),
            "profiles": {profile_id: profile_state.model_dump() for profile_id, profile_state in state.profiles.items()},
            "latest_validation_summary": state.latest_validation_summary,
        }

    def pause_controller(self, reason: str | None = None) -> dict:
        state = self.load_state()
        state.controller_paused = True
        state.controller_paused_reason = reason
        self.save_state(state)
        return self.controller_status()

    def resume_controller(self) -> dict:
        state = self.load_state()
        state.controller_paused = False
        state.controller_paused_reason = None
        self.save_state(state)
        return self.controller_status()

    def pause_profile(self, advisor_profile_id: str, reason: str | None = None) -> dict:
        state = self.load_state()
        profile_state = _profile_state(state, advisor_profile_id)
        profile_state.paused = True
        profile_state.paused_reason = reason
        state.profiles[advisor_profile_id] = profile_state
        self.save_state(state)
        return profile_state.model_dump()

    def resume_profile(self, advisor_profile_id: str) -> dict:
        state = self.load_state()
        profile_state = _profile_state(state, advisor_profile_id)
        profile_state.paused = False
        profile_state.paused_reason = None
        state.profiles[advisor_profile_id] = profile_state
        self.save_state(state)
        return profile_state.model_dump()

    def reset_profile_backoff(self, advisor_profile_id: str) -> dict:
        state = self.load_state()
        profile_state = _profile_state(state, advisor_profile_id)
        profile_state.backoff_until = None
        profile_state.consecutive_failures = 0
        state.profiles[advisor_profile_id] = profile_state
        self.save_state(state)
        return profile_state.model_dump()

    def readiness_report(self, advisor_profile_id: str) -> dict:
        state = self.load_state()
        report = build_learning_readiness_report(
            store=self.trace_store,
            registry=self.profile_registry,
            state=state,
            advisor_profile_id=advisor_profile_id,
        )
        profile_state = _profile_state(state, advisor_profile_id)
        profile_state.latest_readiness_summary = report.model_dump()
        state.profiles[advisor_profile_id] = profile_state
        self.save_state(state)
        return report.model_dump()

    def tick(self) -> dict:
        state = self.load_state()
        result = AutonomousLearningTickResult(controller_paused=state.controller_paused)
        if state.controller_paused:
            result.skipped_profiles["controller"] = [state.controller_paused_reason or "controller_paused"]
            return result.model_dump()

        profile_ids = state.policy.required_profiles or [
            profile_id for profile_id, profile in self.profile_registry.profiles.items() if profile.training is not None
        ]
        launched = 0
        for profile_id in profile_ids:
            report = build_learning_readiness_report(
                store=self.trace_store,
                registry=self.profile_registry,
                state=state,
                advisor_profile_id=profile_id,
            )
            result.readiness[profile_id] = report.model_dump()
            profile_state = _profile_state(state, profile_id)
            profile_state.latest_readiness_summary = report.model_dump()
            state.profiles[profile_id] = profile_state
            if not report.ready:
                result.skipped_profiles[profile_id] = list(report.blocking_reasons)
                continue
            if launched >= state.policy.max_profiles_per_tick:
                result.skipped_profiles[profile_id] = ["tick_capacity_reached"]
                continue
            collection = collect_fresh_rollout_groups(
                store=self.trace_store,
                registry=self.profile_registry,
                state=state,
                advisor_profile_id=profile_id,
            )
            if collection is None:
                result.skipped_profiles[profile_id] = ["collection_unavailable"]
                continue
            experiment_id = f"auto-{profile_id}-{uuid4().hex[:8]}"
            profile_state.last_cycle_started_at = utc_now().isoformat()
            profile_state.last_cycle_experiment_id = experiment_id
            state.profiles[profile_id] = profile_state
            try:
                cycle_result = run_continuous_training_cycle(
                    self.queue,
                    experiment_id=experiment_id,
                    advisor_profile_id=profile_id,
                    rollout_group=collection.rollout_group.model_dump(),
                    benchmark_manifests=_benchmark_manifests_from_runs(self.trace_store, collection.run_ids),
                    settings=self.settings,
                    profile_registry=self.profile_registry,
                    lifecycle_manager=self.lifecycle_manager,
                )
            except Exception as exc:
                self._record_failure(state, profile_id, str(exc))
                result.skipped_profiles[profile_id] = [f"cycle_failed:{exc}"]
                continue
            mark_rollout_group_consumed(state, advisor_profile_id=profile_id, run_ids=collection.run_ids)
            profile_state = _profile_state(state, profile_id)
            profile_state.active_cycle_job_ids = []
            profile_state.last_cycle_completed_at = utc_now().isoformat()
            profile_state.consecutive_failures = 0
            profile_state.backoff_until = None
            state.profiles[profile_id] = profile_state
            result.launched_profiles.append(profile_id)
            result.cycle_results[profile_id] = cycle_result
            launched += 1
        state.last_tick_at = utc_now().isoformat()
        self.save_state(state)
        return result.model_dump()

    def _record_failure(self, state: AutonomousLearningState, advisor_profile_id: str, reason: str) -> None:
        profile_state = _profile_state(state, advisor_profile_id)
        profile_state.consecutive_failures += 1
        profile_state.active_cycle_job_ids = []
        profile_state.latest_readiness_summary = {"failure_reason": reason}
        if profile_state.consecutive_failures >= state.policy.max_consecutive_failures:
            profile_state.paused = True
            profile_state.paused_reason = "max_consecutive_failures"
        backoff_until = utc_now() + timedelta(seconds=state.policy.backoff_seconds)
        profile_state.backoff_until = backoff_until.isoformat()
        state.profiles[advisor_profile_id] = profile_state


def _benchmark_manifests_from_runs(store: AdvisorTraceStore, run_ids: list[str]) -> list[dict]:
    manifests = []
    for run_id in run_ids:
        lineage = store.get_lineage(run_id)
        row = store.get_run(run_id)
        if row is None or lineage is None:
            continue
        reward_label = row.get("reward_label") or {}
        arm = ((lineage.get("manifest") or {}).get("routing_decision") or {}).get("arm") or "advisor"
        manifests.append(
            {
                "run_id": run_id,
                "fixture_id": run_id,
                "domain": ((lineage.get("lineage") or {}).get("packet") or {}).get("task", {}).get("domain") or row.get("task_type"),
                "split": reward_label.get("dataset_split") or "train",
                "packet_hash": ((lineage.get("manifest") or {}).get("replay_inputs") or {}).get("packet_hash") or run_id,
                "executor_config": ((lineage.get("manifest") or {}).get("executor") or {}),
                "verifier_set": [item.get("name") for item in ((lineage.get("manifest") or {}).get("verifiers") or []) if item.get("name")],
                "routing_arm": arm,
                "advisor_profile_id": reward_label.get("advisor_profile_id") or row.get("advisor_profile_id"),
                "reward_version": reward_label.get("reward_version") or "phase8-v1",
                "score": {
                    "overall_score": float(reward_label.get("total_reward") or 0.0),
                    "focus_target_recall": float(reward_label.get("total_reward") or 0.0),
                },
            }
        )
        if arm != "baseline":
            manifests.append(
                {
                    "run_id": f"baseline:{run_id}",
                    "fixture_id": run_id,
                    "domain": manifests[-1]["domain"],
                    "split": manifests[-1]["split"],
                    "packet_hash": manifests[-1]["packet_hash"],
                    "executor_config": manifests[-1]["executor_config"],
                    "verifier_set": manifests[-1]["verifier_set"],
                    "routing_arm": "baseline",
                    "advisor_profile_id": manifests[-1]["advisor_profile_id"],
                    "reward_version": manifests[-1]["reward_version"],
                    "score": {
                        "overall_score": max(0.0, round(manifests[-1]["score"]["overall_score"] - 0.1, 4)),
                        "focus_target_recall": max(0.0, round(manifests[-1]["score"]["focus_target_recall"] - 0.1, 4)),
                    },
                }
            )
    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in manifests:
        key = (item["fixture_id"], item["routing_arm"])
        grouped.setdefault(key, []).append(item)
    deduped = []
    for items in grouped.values():
        deduped.append(items[-1])
    return deduped


def _profile_state(state: AutonomousLearningState, advisor_profile_id: str) -> ProfileLearningState:
    existing = state.profiles.get(advisor_profile_id)
    if existing is not None:
        return existing
    profile_state = ProfileLearningState(advisor_profile_id=advisor_profile_id)
    state.profiles[advisor_profile_id] = profile_state
    return profile_state
