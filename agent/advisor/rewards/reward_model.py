from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, RewardComponentScores, RewardLabel


@dataclass(frozen=True)
class RewardWeights:
    task_success: float = 0.35
    efficiency: float = 0.15
    targeting_quality: float = 0.2
    constraint_compliance: float = 0.2
    human_usefulness: float = 0.1

    def normalized_items(self) -> dict[str, float]:
        raw = {
            "task_success": max(self.task_success, 0.0),
            "efficiency": max(self.efficiency, 0.0),
            "targeting_quality": max(self.targeting_quality, 0.0),
            "constraint_compliance": max(self.constraint_compliance, 0.0),
            "human_usefulness": max(self.human_usefulness, 0.0),
        }
        total = sum(raw.values()) or 1.0
        return {key: value / total for key, value in raw.items()}


def compute_reward_label(
    packet: AdvisorInputPacket,
    advice: AdviceBlock,
    outcome: AdvisorOutcome,
    *,
    human_rating: float | None = None,
    constraint_violations: list[str] | None = None,
    weights: RewardWeights | None = None,
) -> RewardLabel:
    weights = weights or RewardWeights()
    normalized = weights.normalized_items()
    components = RewardComponentScores(
        task_success=_score_task_success(outcome),
        efficiency=_score_efficiency(outcome),
        targeting_quality=_score_targeting_quality(advice, outcome),
        constraint_compliance=_score_constraint_compliance(packet, constraint_violations),
        human_usefulness=_score_human_usefulness(human_rating),
    )
    total_reward = sum(getattr(components, name) * weight for name, weight in normalized.items())
    total_reward = _clamp(total_reward)
    notes = _build_notes(outcome, constraint_violations)
    return RewardLabel(
        run_id=outcome.run_id,
        advisor_profile_id="legacy-default",
        reward_profile_id="legacy-generic",
        reward_formula="weighted_components",
        reward_version="phase8-v1",
        raw_reward=total_reward,
        total_reward=total_reward,
        quality_score=total_reward,
        reward_diagnostics=components.model_dump(),
        components=components,
        dataset_split=_assign_dataset_split(packet),
        example_type=_classify_example_type(outcome, total_reward),
        hard_case_bucket=_assign_hard_case_bucket(components, outcome, constraint_violations),
        notes=notes,
    )


def _score_task_success(outcome: AdvisorOutcome) -> float:
    return {
        "success": 1.0,
        "partial": 0.5,
        "failure": 0.0,
    }.get(outcome.status, 0.0)


def _score_efficiency(outcome: AdvisorOutcome) -> float:
    retry_penalty = min(max(outcome.retries, 0), 4) / 4.0
    return _clamp(1.0 - retry_penalty)


def _score_targeting_quality(advice: AdviceBlock, outcome: AdvisorOutcome) -> float:
    touched = {item for item in outcome.files_touched if item}
    if not touched:
        return 0.0
    focus_targets = {item.locator for item in advice.focus_targets if item.locator}
    relevant_files = {item.path for item in advice.relevant_files if item.path}
    advised_targets = focus_targets | relevant_files
    if not advised_targets:
        return 0.0
    return 1.0 if touched & advised_targets else 0.0


def _score_constraint_compliance(packet: AdvisorInputPacket, constraint_violations: list[str] | None) -> float:
    violations = [item for item in (constraint_violations or []) if item]
    if not violations:
        return 1.0
    baseline = max(len(packet.constraints), len(violations), 1)
    return _clamp(1.0 - (len(violations) / baseline))


def _score_human_usefulness(human_rating: float | None) -> float:
    if human_rating is None:
        return 0.5
    return _clamp(human_rating / 5.0)


def _assign_dataset_split(packet: AdvisorInputPacket) -> str:
    repo_family = packet.repo.get("path", "")
    task_family = packet.task.type if packet.task else packet.task_type
    digest = hashlib.sha256(f"{repo_family}|{task_family}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket <= 2:
        return "val"
    return "train"


def _classify_example_type(outcome: AdvisorOutcome, total_reward: float) -> str:
    if outcome.status == "failure" or total_reward < 0.25:
        return "negative"
    if outcome.status == "success" and total_reward >= 0.75:
        return "positive"
    return "neutral"


def _assign_hard_case_bucket(
    components: RewardComponentScores,
    outcome: AdvisorOutcome,
    constraint_violations: list[str] | None,
) -> str | None:
    if constraint_violations:
        return "constraint_failure"
    if outcome.status == "failure":
        return "failed_execution"
    if components.targeting_quality < 0.5:
        return "targeting_miss"
    if components.efficiency < 0.5:
        return "inefficient_execution"
    return None


def _build_notes(outcome: AdvisorOutcome, constraint_violations: list[str] | None) -> list[str]:
    notes: list[str] = []
    if outcome.review_verdict:
        notes.append(outcome.review_verdict)
    notes.extend(item for item in (constraint_violations or []) if item)
    return notes


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))
