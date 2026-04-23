from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .orchestration import (
    ExecutorRequest,
    ExecutorRunResult,
    RoutingDecision,
    VerifierRunRecord,
)
from .reward_registry import RewardRegistry
from .schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, RewardLabel


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
    advice = AdviceBlock.model_validate(runtime.generate_advice(request.packet, system_prompt=request.system_prompt))
    executor_request = ExecutorRequest(
        run_id=request.packet.run_id,
        packet=request.packet,
        advice=advice,
        rendered_advice=None,
        routing_decision=RoutingDecision(
            arm="advisor",
            advisor_fraction=1.0,
            routing_key=request.rollout_id,
            bucket=0.0,
        ),
    )
    executor_result = ExecutorRunResult.model_validate(executor.execute(executor_request))
    verifier_records = [
        VerifierRunRecord(descriptor=verifier.descriptor, result=verifier.verify(executor_request, executor_result))
        for verifier in verifiers
    ]
    outcome = _build_outcome(request.packet.run_id, executor_result, verifier_records)
    constraint_violations = [
        item
        for record in verifier_records
        for item in record.result.constraint_violations
        if item
    ]
    reward_label = reward_registry.compute_for_profile_id(
        request.advisor_profile_id,
        request.packet,
        advice,
        outcome,
        executor_result=executor_result.model_dump(),
        verifier_results=[record.model_dump() for record in verifier_records],
        constraint_violations=constraint_violations,
    )
    transcript = list(request.multi_turn_transcript)
    return TrainingRolloutResult(
        rollout_id=request.rollout_id,
        advisor_profile_id=request.advisor_profile_id,
        packet=request.packet,
        primary_advice=advice,
        executor_result=executor_result,
        verifier_results=verifier_records,
        outcome=outcome,
        reward_label=reward_label,
        diagnostics={
            "verifier_count": len(verifier_records),
            "constraint_violations": constraint_violations,
            "executor_kind": request.executor_kind,
            "multi_turn": bool(transcript),
        },
        multi_turn_transcript=transcript,
    )


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
