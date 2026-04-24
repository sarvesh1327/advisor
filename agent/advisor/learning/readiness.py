from __future__ import annotations

from statistics import pstdev
from typing import Any

from pydantic import BaseModel, Field

from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.storage.trace_store import AdvisorTraceStore
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult, TrainingRolloutResult

from .state import AutonomousLearningPolicy, AutonomousLearningState, ProfileLearningState, parse_ts, utc_now


class LearningReadinessReport(BaseModel):
    advisor_profile_id: str
    ready: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    fresh_run_count: int = 0
    fresh_trajectory_count: int = 0
    distinct_reward_count: int = 0
    latest_run_ids: list[str] = Field(default_factory=list)
    latest_trajectory_ids: list[str] = Field(default_factory=list)
    reward_values: list[float] = Field(default_factory=list)
    active_cycle_job_ids: list[str] = Field(default_factory=list)
    backoff_until: str | None = None
    paused: bool = False
    summary: dict[str, Any] = Field(default_factory=dict)


class FreshRolloutCollection(BaseModel):
    advisor_profile_id: str
    run_ids: list[str] = Field(default_factory=list)
    trajectory_ids: list[str] = Field(default_factory=list)
    rollout_group: TrainingRolloutGroupResult


class TrainingRunIngestionRecord(BaseModel):
    run_id: str
    trajectory_id: str
    advisor_profile_id: str
    reward_value: float
    packet: dict
    advice: dict
    outcome: dict
    reward_label: dict
    lineage: dict
    trajectory: dict


def build_learning_readiness_report(
    *,
    store: AdvisorTraceStore,
    registry: AdvisorProfileRegistry,
    state: AutonomousLearningState,
    advisor_profile_id: str,
    policy: AutonomousLearningPolicy | None = None,
) -> LearningReadinessReport:
    active_policy = policy or state.policy
    profile = registry.resolve(advisor_profile_id)
    profile_state = _profile_state(state, advisor_profile_id)
    fresh_run_records = list(
        _fresh_rewarded_run_records(store=store, profile_state=profile_state, advisor_profile_id=advisor_profile_id)
    )
    fresh_records = list(_fresh_training_records(store=store, profile_state=profile_state, advisor_profile_id=advisor_profile_id))
    reward_values = [record.reward_value for record in fresh_records]
    distinct_reward_count = len({round(value, 6) for value in reward_values})
    required_rollout_group_size = profile.training.rollout_group_size if profile.training is not None else None
    blocking_reasons: list[str] = []
    if profile.training is None:
        blocking_reasons.append("training_not_configured")
    elif profile.training.rollout_group_size <= 0:
        blocking_reasons.append("invalid_rollout_group_size")
    if profile_state.paused:
        blocking_reasons.append("profile_paused")
    backoff_until = parse_ts(profile_state.backoff_until)
    if backoff_until and backoff_until > utc_now():
        blocking_reasons.append("profile_in_backoff")
    if profile_state.active_cycle_job_ids:
        blocking_reasons.append("cycle_already_active")
    if len(fresh_run_records) < active_policy.min_fresh_runs:
        blocking_reasons.append("insufficient_fresh_runs")
    if required_rollout_group_size is not None and required_rollout_group_size > 0:
        if len(fresh_records) < required_rollout_group_size:
            blocking_reasons.append("insufficient_fresh_trajectories")
    if distinct_reward_count < active_policy.min_distinct_rewards:
        blocking_reasons.append("insufficient_reward_variation")
    last_completed = parse_ts(profile_state.last_cycle_completed_at)
    if last_completed is not None and active_policy.min_seconds_between_cycles > 0:
        elapsed = (utc_now() - last_completed).total_seconds()
        if elapsed < active_policy.min_seconds_between_cycles:
            blocking_reasons.append("cooldown_active")
    return LearningReadinessReport(
        advisor_profile_id=advisor_profile_id,
        ready=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        fresh_run_count=len(fresh_run_records),
        fresh_trajectory_count=len(fresh_records),
        distinct_reward_count=distinct_reward_count,
        latest_run_ids=[record.run_id for record in fresh_records],
        latest_trajectory_ids=[record.trajectory_id for record in fresh_records],
        reward_values=reward_values,
        active_cycle_job_ids=list(profile_state.active_cycle_job_ids),
        backoff_until=profile_state.backoff_until,
        paused=profile_state.paused,
        summary={
            "reward_stddev": round(float(pstdev(reward_values)), 6) if len(reward_values) > 1 else 0.0,
            "required_rollout_group_size": required_rollout_group_size,
            "consumed_run_count": len(profile_state.consumed_run_ids),
            "consumed_trajectory_count": len(profile_state.consumed_trajectory_ids),
        },
    )


def collect_fresh_rollout_groups(
    *,
    store: AdvisorTraceStore,
    registry: AdvisorProfileRegistry,
    state: AutonomousLearningState,
    advisor_profile_id: str,
    policy: AutonomousLearningPolicy | None = None,
) -> FreshRolloutCollection | None:
    report = build_learning_readiness_report(
        store=store,
        registry=registry,
        state=state,
        advisor_profile_id=advisor_profile_id,
        policy=policy,
    )
    if not report.ready:
        return None
    profile = registry.resolve(advisor_profile_id)
    profile_state = _profile_state(state, advisor_profile_id)
    fresh_records = list(_fresh_training_records(store=store, profile_state=profile_state, advisor_profile_id=advisor_profile_id))
    limit = profile.training.rollout_group_size if profile.training is not None else len(fresh_records)
    selected = fresh_records[:limit]
    if not selected:
        return None
    group_id = f"auto-group:{advisor_profile_id}:{selected[-1].trajectory_id}"
    rollout_group = TrainingRolloutGroupResult(
        group_id=group_id,
        advisor_profile_id=advisor_profile_id,
        results=[
            TrainingRolloutResult(
                rollout_id=f"rollout:{record.trajectory_id}",
                advisor_profile_id=advisor_profile_id,
                packet=record.packet,
                primary_advice=record.advice,
                executor_result=(record.lineage.get("lineage") or {}).get("executor_result") or {},
                verifier_results=(record.lineage.get("lineage") or {}).get("verifier_results") or [],
                outcome=record.outcome,
                reward_label=record.reward_label,
                diagnostics={
                    "source_run_id": record.run_id,
                    "source_trajectory_id": record.trajectory_id,
                    "dogfood": True,
                },
                trajectory=record.trajectory,
            )
            for record in selected
        ],
        reward_values=[record.reward_value for record in selected],
        summary={
            "mean_reward": round(sum(record.reward_value for record in selected) / len(selected), 4),
            "source_run_ids": [record.run_id for record in selected],
            "source_trajectory_ids": [record.trajectory_id for record in selected],
            "dogfood": any((record.packet.get("repo") or {}).get("path", "").endswith("Advisor") for record in selected),
        },
    )
    return FreshRolloutCollection(
        advisor_profile_id=advisor_profile_id,
        run_ids=[record.run_id for record in selected],
        trajectory_ids=[record.trajectory_id for record in selected],
        rollout_group=rollout_group,
    )


def mark_rollout_group_consumed(
    state: AutonomousLearningState,
    *,
    advisor_profile_id: str,
    run_ids: list[str],
    trajectory_ids: list[str] | None = None,
) -> AutonomousLearningState:
    profile_state = _profile_state(state, advisor_profile_id)
    seen_runs = list(profile_state.consumed_run_ids)
    for run_id in run_ids:
        if run_id not in seen_runs:
            seen_runs.append(run_id)
    seen_trajectories = list(profile_state.consumed_trajectory_ids)
    for trajectory_id in trajectory_ids or []:
        if trajectory_id not in seen_trajectories:
            seen_trajectories.append(trajectory_id)
    profile_state.consumed_run_ids = seen_runs
    profile_state.consumed_trajectory_ids = seen_trajectories
    state.profiles[advisor_profile_id] = profile_state
    return state


def _fresh_training_records(
    *,
    store: AdvisorTraceStore,
    profile_state: ProfileLearningState,
    advisor_profile_id: str,
):
    consumed_runs = set(profile_state.consumed_run_ids)
    consumed_trajectories = set(profile_state.consumed_trajectory_ids)
    rows = {
        row["run_id"]: row
        for row in store.list_runs(include_context=True)
        if row.get("advisor_profile_id") == advisor_profile_id
    }
    for trajectory in store.list_trajectories():
        trajectory_id = trajectory.get("trajectory_id")
        run_id = trajectory.get("run_id")
        if trajectory.get("advisor_profile_id") != advisor_profile_id:
            continue
        if trajectory_id in consumed_trajectories or run_id in consumed_runs:
            continue
        final_reward = trajectory.get("final_reward")
        final_outcome = trajectory.get("final_outcome") or {}
        if not _trajectory_has_final_reward(final_reward):
            continue
        if final_outcome.get("status") is None:
            continue
        row = rows.get(run_id)
        if row is None:
            continue
        canonical_reward = row.get("reward_label") or {}
        if canonical_reward.get("total_reward") is None:
            continue
        lineage = store.get_lineage(run_id)
        if lineage is None:
            continue
        yield TrainingRunIngestionRecord(
            run_id=run_id,
            trajectory_id=trajectory_id,
            advisor_profile_id=advisor_profile_id,
            reward_value=float(canonical_reward.get("total_reward") or 0.0),
            packet=trajectory["turns"][0].get("state_packet") if trajectory.get("turns") else row.get("input") or {},
            advice=trajectory["turns"][0].get("advice") if trajectory.get("turns") else row.get("advice") or {},
            outcome=final_outcome,
            reward_label=canonical_reward,
            lineage=lineage,
            trajectory=trajectory,
        )


def _trajectory_has_final_reward(final_reward) -> bool:
    if isinstance(final_reward, dict):
        return final_reward.get("total_reward") is not None
    return isinstance(final_reward, (int, float)) and not isinstance(final_reward, bool)



def _fresh_rewarded_run_records(
    *,
    store: AdvisorTraceStore,
    profile_state: ProfileLearningState,
    advisor_profile_id: str,
):
    consumed = set(profile_state.consumed_run_ids)
    rows = list(reversed(store.list_runs(include_context=True)))
    for row in rows:
        if row.get("advisor_profile_id") != advisor_profile_id:
            continue
        if row.get("run_id") in consumed:
            continue
        reward_label = row.get("reward_label") or {}
        outcome = row.get("outcome") or {}
        if not reward_label or not outcome:
            continue
        lineage = store.get_lineage(row["run_id"])
        if lineage is None:
            continue
        if outcome.get("status") is None:
            continue
        yield row


def _profile_state(state: AutonomousLearningState, advisor_profile_id: str) -> ProfileLearningState:
    existing = state.profiles.get(advisor_profile_id)
    if existing is not None:
        return existing
    profile_state = ProfileLearningState(advisor_profile_id=advisor_profile_id)
    state.profiles[advisor_profile_id] = profile_state
    return profile_state
