from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel

from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult


# Phase 6 keeps rollout quality checks reusable and explicit before train/promotion side effects.
def build_phase6_hardening_report(
    *,
    rollout_group: TrainingRolloutGroupResult | dict,
    advisor_profile_id: str | None = None,
) -> dict:
    group = TrainingRolloutGroupResult.model_validate(rollout_group)
    issues: list[dict[str, Any]] = []

    expected_profile_id = advisor_profile_id or group.advisor_profile_id
    if group.advisor_profile_id != expected_profile_id:
        issues.append(
            _issue(
                code="rollout_group_profile_mismatch",
                message=(
                    f"rollout group advisor_profile_id {group.advisor_profile_id} does not match expected {expected_profile_id}"
                ),
            )
        )

    if len(group.reward_values) != group.rollout_count:
        issues.append(
            _issue(
                code="reward_value_count_mismatch",
                message=(
                    f"rollout group reward_values count {len(group.reward_values)} does not match rollout_count {group.rollout_count}"
                ),
            )
        )

    signature_to_rollouts: dict[str, list[str]] = {}
    normalized_rewards: list[float] = []
    for index, result in enumerate(group.results):
        if result.advisor_profile_id != group.advisor_profile_id:
            issues.append(
                _issue(
                    code="rollout_profile_mismatch",
                    message=(
                        f"rollout {result.rollout_id} advisor_profile_id {result.advisor_profile_id} does not match group {group.advisor_profile_id}"
                    ),
                    rollout_id=result.rollout_id,
                )
            )

        reward_value = _reward_total(result.reward_label)
        normalized_rewards.append(reward_value)
        if not math.isfinite(reward_value) or reward_value < 0.0 or reward_value > 1.0:
            issues.append(
                _issue(
                    code="invalid_reward_value",
                    message=(
                        f"rollout {result.rollout_id} reward {reward_value} must be finite and within [0.0, 1.0]"
                    ),
                    rollout_id=result.rollout_id,
                )
            )

        if index < len(group.reward_values):
            group_reward_value = float(group.reward_values[index])
            if not math.isfinite(group_reward_value) or abs(group_reward_value - reward_value) > 1e-6:
                issues.append(
                    _issue(
                        code="reward_value_mismatch",
                        message=(
                            f"rollout {result.rollout_id} reward_label total {reward_value} does not match reward_values[{index}]={group_reward_value}"
                        ),
                        rollout_id=result.rollout_id,
                    )
                )

        signature = _training_signature(group.advisor_profile_id, result.packet, result.primary_advice)
        signature_to_rollouts.setdefault(signature, []).append(result.rollout_id)

    duplicate_signature_count = 0
    for signature, rollout_ids in signature_to_rollouts.items():
        if len(rollout_ids) <= 1:
            continue
        duplicate_signature_count += 1
        issues.append(
            _issue(
                code="duplicate_training_signature",
                message=(
                    f"rollout group contains duplicate packet/advice training signatures for rollouts {', '.join(rollout_ids)}"
                ),
                sample_signature=signature,
            )
        )

    distinct_reward_count = len({round(value, 6) for value in normalized_rewards})
    if group.rollout_count > 1 and normalized_rewards and distinct_reward_count <= 1:
        issues.append(
            _issue(
                code="flat_reward_distribution",
                message="rollout group rewards are flat, so the group provides no relative learning signal",
            )
        )

    return {
        "advisor_profile_id": expected_profile_id,
        "blocking": bool(issues),
        "issues": issues,
        "summary": {
            "rollout_count": group.rollout_count,
            "reward_value_count": len(group.reward_values),
            "distinct_reward_count": distinct_reward_count,
            "duplicate_signature_count": duplicate_signature_count,
            "reward_min": min(normalized_rewards) if normalized_rewards else None,
            "reward_max": max(normalized_rewards) if normalized_rewards else None,
        },
    }


# Promotion fallback should block bad evidence instead of letting malformed/regressive payloads promote.
def build_phase6_promotion_guard(
    *,
    evaluation: dict,
    advisor_profile_id: str,
    candidate_checkpoint_id: str,
) -> dict | None:
    if evaluation.get("advisor_profile_id") != advisor_profile_id:
        return _blocked_promotion(
            advisor_profile_id=advisor_profile_id,
            checkpoint_id=candidate_checkpoint_id,
            reason="promotion evaluation advisor_profile_id does not match payload",
        )
    if evaluation.get("candidate_checkpoint_id") != candidate_checkpoint_id:
        return _blocked_promotion(
            advisor_profile_id=advisor_profile_id,
            checkpoint_id=candidate_checkpoint_id,
            reason="promotion evaluation candidate_checkpoint_id does not match payload",
        )
    if evaluation.get("rollback") is True:
        return _blocked_promotion(
            advisor_profile_id=advisor_profile_id,
            checkpoint_id=candidate_checkpoint_id,
            reason=evaluation.get("decision_reason") or "promotion blocked because evaluation indicates rollback",
        )

    deltas = dict(evaluation.get("deltas") or {})
    if not deltas:
        return None
    overall_delta = deltas.get("overall_score")
    recall_delta = deltas.get("focus_target_recall")
    if not _is_finite_number(overall_delta) or not _is_finite_number(recall_delta):
        return _blocked_promotion(
            advisor_profile_id=advisor_profile_id,
            checkpoint_id=candidate_checkpoint_id,
            reason="promotion blocked because evaluation deltas are missing or non-finite",
        )
    if float(overall_delta) <= 0.0 or float(recall_delta) < 0.0:
        return _blocked_promotion(
            advisor_profile_id=advisor_profile_id,
            checkpoint_id=candidate_checkpoint_id,
            reason=evaluation.get("decision_reason") or "promotion blocked because evaluation deltas are regressive",
        )
    return None



def _issue(*, code: str, message: str, rollout_id: str | None = None, sample_signature: str | None = None) -> dict:
    payload = {
        "code": code,
        "severity": "blocking",
        "message": message,
    }
    if rollout_id is not None:
        payload["rollout_id"] = rollout_id
    if sample_signature is not None:
        payload["sample_signature"] = sample_signature
    return payload



def _training_signature(profile_id: str, packet: BaseModel | dict, advice: BaseModel | dict) -> str:
    normalized = json.dumps(
        {
            "advisor_profile_id": profile_id,
            "packet": _normalized_signature_payload(packet),
            "advice": _normalized_signature_payload(advice),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]



def _normalized_signature_payload(payload: BaseModel | dict) -> dict:
    return _strip_signature_noise(_normalized_payload(payload))



def _strip_signature_noise(value: Any):
    if isinstance(value, dict):
        return {
            key: _strip_signature_noise(item)
            for key, item in value.items()
            if key not in {"run_id", "session_id"}
        }
    if isinstance(value, list):
        return [_strip_signature_noise(item) for item in value]
    return value



def _normalized_payload(payload: BaseModel | dict) -> dict:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    return dict(payload)



def _reward_total(reward_label: BaseModel | dict) -> float:
    if isinstance(reward_label, BaseModel):
        return float(reward_label.model_dump().get("total_reward", 0.0))
    return float(dict(reward_label).get("total_reward", 0.0))



def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False



def _blocked_promotion(*, advisor_profile_id: str, checkpoint_id: str, reason: str) -> dict:
    return {
        "advisor_profile_id": advisor_profile_id,
        "checkpoint_id": checkpoint_id,
        "promoted": False,
        "status": "blocked",
        "reason": reason,
    }
