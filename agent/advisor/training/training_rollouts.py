from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel, Field

from agent.advisor.core.schemas import (
    AdviceBlock,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTrajectory,
    AdvisorTrajectoryTurn,
    RewardLabel,
    TurnObservation,
)
from agent.advisor.execution.orchestration import (
    ExecutorRequest,
    ExecutorRunResult,
    ExecutorStepRequest,
    ExecutorStepResult,
    RoutingDecision,
    VerifierRunRecord,
    run_executor_step,
)
from agent.advisor.rewards.reward_registry import RewardRegistry


class RolloutTurnRecord(BaseModel):
    turn_index: int
    actor: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingRolloutRequest(BaseModel):
    rollout_id: str
    advisor_profile_id: str
    packet: AdvisorInputPacket
    executor_name: str
    executor_kind: str
    verifier_names: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    max_turns: int = 1
    group_id: str | None = None
    candidate_index: int | None = None
    multi_turn_transcript: list[RolloutTurnRecord] = Field(default_factory=list)


class TrainingRolloutResult(BaseModel):
    rollout_id: str
    advisor_profile_id: str
    packet: AdvisorInputPacket | dict[str, Any]
    primary_advice: AdviceBlock | dict[str, Any]
    executor_result: ExecutorRunResult | dict[str, Any]
    verifier_results: list[VerifierRunRecord | dict[str, Any]] = Field(default_factory=list)
    outcome: AdvisorOutcome | dict[str, Any]
    reward_label: RewardLabel | dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    multi_turn_transcript: list[RolloutTurnRecord] = Field(default_factory=list)
    trajectory: dict[str, Any] = Field(default_factory=dict)


class TrainingRolloutGroupRequest(BaseModel):
    group_id: str
    advisor_profile_id: str
    requests: list[TrainingRolloutRequest] = Field(default_factory=list)


class TrainingRolloutGroupResult(BaseModel):
    group_id: str
    advisor_profile_id: str
    results: list[TrainingRolloutResult] = Field(default_factory=list)
    reward_values: list[float] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def rollout_count(self) -> int:
        return len(self.results)


def execute_training_rollout(
    request: TrainingRolloutRequest,
    *,
    runtime: Any,
    executor: Any,
    verifiers: list[Any],
    reward_registry: RewardRegistry,
) -> TrainingRolloutResult:
    current_packet = request.packet
    routing_decision = RoutingDecision(
        arm="advisor",
        advisor_fraction=1.0,
        routing_key=request.rollout_id,
        bucket=0.0,
    )
    observations: list[TurnObservation] = []
    trajectory_turns: list[AdvisorTrajectoryTurn] = []
    verifier_records: list[VerifierRunRecord] = []
    final_executor_result: ExecutorRunResult | None = None
    primary_advice: AdviceBlock | None = None
    stop_reason = "max_turns"

    for turn_index in range(max(1, request.max_turns)):
        advice = _generate_rollout_advice(runtime, current_packet, request)
        if primary_advice is None:
            primary_advice = advice
        step_request = ExecutorStepRequest(
            trajectory_id=request.rollout_id,
            turn_index=turn_index,
            packet=current_packet,
            advice=advice,
            previous_observations=list(observations),
            budget={"max_turns": request.max_turns},
            routing_decision=routing_decision,
        )
        step_result = run_executor_step(executor, step_request)
        executor_result = _step_result_to_executor_result(step_result)
        final_executor_result = executor_result
        executor_request = ExecutorRequest(
            run_id=current_packet.run_id,
            packet=current_packet,
            advice=advice,
            rendered_advice=None,
            routing_decision=routing_decision,
        )
        verifier_records = [
            VerifierRunRecord(descriptor=verifier.descriptor, result=verifier.verify(executor_request, executor_result))
            for verifier in verifiers
        ]
        observation = _observation_from_step_result(turn_index, step_result, verifier_records)
        observations.append(observation)
        trajectory_turns.append(
            AdvisorTrajectoryTurn(
                turn_index=turn_index,
                state_packet=current_packet,
                advice=advice,
                observation=observation,
            )
        )
        stop_reason = _rollout_stop_reason(step_result, turn_index, request.max_turns)
        if stop_reason != "continue":
            break
        current_packet = _packet_with_observation(current_packet, observation)

    assert primary_advice is not None
    assert final_executor_result is not None
    final_executor_result = _executor_result_with_turn_metrics(final_executor_result, turn_count=len(trajectory_turns))
    outcome = _build_outcome(request.packet.run_id, final_executor_result, verifier_records)
    constraint_violations = [
        item
        for record in verifier_records
        for item in record.result.constraint_violations
        if item
    ]
    reward_label = reward_registry.compute_for_profile_id(
        request.advisor_profile_id,
        request.packet,
        primary_advice,
        outcome,
        executor_result=final_executor_result.model_dump(),
        verifier_results=[record.model_dump() for record in verifier_records],
        constraint_violations=constraint_violations,
    )
    trajectory = AdvisorTrajectory(
        trajectory_id=request.rollout_id,
        run_id=request.packet.run_id,
        advisor_profile_id=request.advisor_profile_id,
        task_text=request.packet.task_text,
        turns=trajectory_turns,
        final_outcome=outcome,
        final_reward=reward_label,
        stop_reason=stop_reason,
        budget={"max_turns": request.max_turns},
    )
    transcript = list(request.multi_turn_transcript)
    return TrainingRolloutResult(
        rollout_id=request.rollout_id,
        advisor_profile_id=request.advisor_profile_id,
        packet=request.packet,
        primary_advice=primary_advice,
        executor_result=final_executor_result,
        verifier_results=verifier_records,
        outcome=outcome,
        reward_label=reward_label,
        diagnostics={
            "verifier_count": len(verifier_records),
            "constraint_violations": constraint_violations,
            "executor_kind": request.executor_kind,
            "multi_turn": bool(transcript) or len(trajectory_turns) > 1,
            "turn_count": len(trajectory_turns),
            "stop_reason": stop_reason,
        },
        multi_turn_transcript=transcript,
        trajectory=trajectory.model_dump(),
    )


def _generate_rollout_advice(runtime: Any, packet: AdvisorInputPacket, request: TrainingRolloutRequest) -> AdviceBlock:
    generate_advice = runtime.generate_advice
    # Keep rollout runtime calls profile-aware without requiring every runtime to accept the new arg.
    signature = inspect.signature(generate_advice)
    kwargs = {}
    if "system_prompt" in signature.parameters:
        kwargs["system_prompt"] = request.system_prompt
    if "advisor_profile_id" in signature.parameters:
        kwargs["advisor_profile_id"] = request.advisor_profile_id
    return AdviceBlock.model_validate(generate_advice(packet, **kwargs))


def _step_result_to_executor_result(step_result: ExecutorStepResult) -> ExecutorRunResult:
    # Verifiers/rewards still consume the existing executor-result shape.
    retries = _safe_int(step_result.metrics.get("retries"), default=0)
    return ExecutorRunResult(
        status=step_result.status,
        summary=step_result.summary,
        output=step_result.output,
        files_touched=step_result.files_touched,
        tests_run=step_result.tests_run,
        metadata=dict(step_result.metrics),
        retries=retries,
    )


def _executor_result_with_turn_metrics(result: ExecutorRunResult, *, turn_count: int) -> ExecutorRunResult:
    # Missing step metrics should reflect the actual rollout length, not a one-step solve.
    metadata = dict(result.metadata)
    if "steps" not in metadata:
        metadata["steps"] = max(1, turn_count)
    retries = result.retries
    if "retries" not in metadata:
        retries = max(0, turn_count - 1)
        metadata["retries"] = retries
    return result.model_copy(update={"metadata": metadata, "retries": retries})


def _observation_from_step_result(
    turn_index: int,
    step_result: ExecutorStepResult,
    verifier_records: list[VerifierRunRecord],
) -> TurnObservation:
    # Observations are the compact state carried into the next advisor turn.
    verifier_hints = [record.result.summary for record in verifier_records if record.result.summary]
    return TurnObservation(
        turn_index=turn_index,
        status=step_result.status,
        executor_output=step_result.output,
        summary=step_result.summary,
        files_touched=step_result.files_touched,
        tests_run=step_result.tests_run,
        verifier_hints=verifier_hints,
        error_messages=step_result.error_messages,
        metrics=step_result.metrics,
    )


def _packet_with_observation(packet: AdvisorInputPacket, observation: TurnObservation) -> AdvisorInputPacket:
    next_packet = packet.model_copy(deep=True)
    next_packet.history.append(
        AdvisorHistoryEntry(
            kind="turn-observation",
            summary=observation.summary or observation.status,
            locator=f"turn:{observation.turn_index}",
            metadata=observation.model_dump(),
        )
    )
    return next_packet


def _rollout_stop_reason(step_result: ExecutorStepResult, turn_index: int, max_turns: int) -> str:
    if step_result.status == "success":
        return "success"
    if step_result.status == "failure":
        return "failure"
    if step_result.done:
        return "executor_done"
    if turn_index + 1 >= max(1, max_turns):
        return "max_turns"
    return "continue"


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def execute_training_rollout_group(
    request: TrainingRolloutGroupRequest,
    *,
    runtime: Any,
    executor: Any,
    verifiers: list[Any],
    reward_registry: RewardRegistry,
) -> TrainingRolloutGroupResult:
    results = [
        execute_training_rollout(
            rollout_request,
            runtime=runtime,
            executor=executor,
            verifiers=verifiers,
            reward_registry=reward_registry,
        )
        for rollout_request in request.requests
    ]
    reward_values = [float(result.reward_label.total_reward) for result in results]
    summary = {
        "mean_reward": round(sum(reward_values) / len(reward_values), 4) if reward_values else 0.0,
        "max_reward": max(reward_values) if reward_values else 0.0,
        "min_reward": min(reward_values) if reward_values else 0.0,
    }
    return TrainingRolloutGroupResult(
        group_id=request.group_id,
        advisor_profile_id=request.advisor_profile_id,
        results=results,
        reward_values=reward_values,
        summary=summary,
    )


def _build_outcome(
    run_id: str,
    executor_result: ExecutorRunResult,
    verifier_results: list[VerifierRunRecord],
) -> AdvisorOutcome:
    statuses = {record.result.status for record in verifier_results}
    if executor_result.status == "failure" or "fail" in statuses:
        status = "failure"
        review_verdict = "fail"
    elif executor_result.status == "partial" or "warn" in statuses:
        status = "partial"
        review_verdict = "warn"
    else:
        status = "success"
        review_verdict = "pass"
    summary_parts = [executor_result.summary] if executor_result.summary else []
    summary_parts.extend(record.result.summary for record in verifier_results if record.result.summary)
    return AdvisorOutcome(
        run_id=run_id,
        status=status,
        files_touched=executor_result.files_touched,
        retries=executor_result.retries,
        tests_run=executor_result.tests_run,
        review_verdict=review_verdict,
        summary=" | ".join(summary_parts) if summary_parts else None,
    )
