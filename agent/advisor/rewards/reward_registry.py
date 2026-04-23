from __future__ import annotations

import hashlib
from typing import Any

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, RewardLabel
from agent.advisor.domain_rewards import (
    compute_coding_exact_answer_reward,
    compute_coding_swe_efficiency_reward,
    compute_research_writing_match_reward,
    compute_ui_edit_from_screenshot_reward,
    compute_ui_from_text_layout_reward,
)
from agent.advisor.profiles import AdvisorProfile
from agent.advisor.rewards.reward_specs import RewardSpec, default_reward_specs


class RewardRegistry:
    def __init__(self, specs: dict[str, RewardSpec] | None = None):
        self.specs = dict(specs or default_reward_specs())

    @classmethod
    def default(cls) -> "RewardRegistry":
        return cls()

    def get(self, reward_spec_id: str) -> RewardSpec:
        try:
            return self.specs[reward_spec_id]
        except KeyError as exc:
            raise ValueError(f"unknown reward spec: {reward_spec_id}") from exc

    def resolve_spec_for_profile(self, profile: AdvisorProfile) -> RewardSpec:
        reward_spec_id = profile.reward_spec_id or _default_reward_spec_for_profile(profile.profile_id)
        return self.get(reward_spec_id)

    def compute_for_profile_id(
        self,
        profile_id: str,
        packet: AdvisorInputPacket,
        advice: AdviceBlock,
        outcome: AdvisorOutcome,
        *,
        executor_result: dict[str, Any],
        verifier_results: list[dict[str, Any]],
        profile_domain: str | None = None,
        reward_spec_id: str | None = None,
        constraint_violations: list[str] | None = None,
    ) -> RewardLabel:
        profile = AdvisorProfile(
            profile_id=profile_id,
            domain=profile_domain or packet.task.domain,
            reward_spec_id=reward_spec_id,
        )
        return self.compute_reward_for_profile(
            profile,
            packet,
            advice,
            outcome,
            executor_result=executor_result,
            verifier_results=verifier_results,
            constraint_violations=constraint_violations,
        )

    def compute_reward_for_profile(
        self,
        profile: AdvisorProfile,
        packet: AdvisorInputPacket,
        advice: AdviceBlock,
        outcome: AdvisorOutcome,
        *,
        executor_result: dict[str, Any],
        verifier_results: list[dict[str, Any]],
        constraint_violations: list[str] | None = None,
    ) -> RewardLabel:
        spec = self.resolve_spec_for_profile(profile)
        raw_reward, diagnostics = self._compute_raw_reward(
            spec,
            packet,
            advice,
            outcome,
            executor_result=executor_result,
            verifier_results=verifier_results,
            constraint_violations=constraint_violations or [],
        )
        notes = _build_notes(outcome, constraint_violations)
        total_reward = _clamp(raw_reward)
        return RewardLabel(
            run_id=outcome.run_id,
            advisor_profile_id=profile.profile_id,
            reward_profile_id=spec.reward_spec_id,
            reward_formula=spec.formula_name,
            reward_version=spec.reward_version,
            raw_reward=total_reward,
            total_reward=total_reward,
            quality_score=total_reward,
            reward_diagnostics=diagnostics,
            dataset_split=_assign_dataset_split(packet),
            example_type=_classify_example_type(outcome, total_reward),
            hard_case_bucket=_assign_hard_case_bucket(outcome, constraint_violations, diagnostics),
            notes=notes,
        )

    def _compute_raw_reward(
        self,
        spec: RewardSpec,
        packet: AdvisorInputPacket,
        advice: AdviceBlock,
        outcome: AdvisorOutcome,
        *,
        executor_result: dict[str, Any],
        verifier_results: list[dict[str, Any]],
        constraint_violations: list[str],
    ) -> tuple[float, dict[str, Any]]:
        del packet, advice, constraint_violations
        verifier_metadata = _merge_verifier_metadata(verifier_results)
        executor_metadata = dict(executor_result.get("metadata") or {})
        executor_status = executor_result.get("status") or outcome.status
        if spec.formula_name == "coding_swe_efficiency":
            resolved = outcome.status == "success" and executor_status == "success"
            return compute_coding_swe_efficiency_reward(
                resolved=resolved,
                steps=_safe_step_count(executor_metadata.get("steps"), fallback=max(outcome.retries + 1, 1)),
            )
        if spec.formula_name == "coding_exact_answer":
            exact_correct = bool(verifier_metadata.get("exact_correct") or verifier_metadata.get("exact_match"))
            return compute_coding_exact_answer_reward(exact_correct=exact_correct)
        if spec.formula_name == "ui_from_text_layout":
            return compute_ui_from_text_layout_reward(
                render_valid=bool(executor_metadata.get("render_valid", executor_status != "failure")),
                hard_constraint_pass_rate=float(verifier_metadata.get("hard_constraint_pass_rate") or 0.0),
                soft_style_score=float(verifier_metadata.get("soft_style_score") or verifier_metadata.get("style_score") or 0.0),
            )
        if spec.formula_name == "ui_edit_from_screenshot":
            return compute_ui_edit_from_screenshot_reward(
                render_valid=bool(executor_metadata.get("render_valid", executor_status != "failure")),
                screenshot_similarity=_metadata_float(verifier_metadata, "screenshot_similarity"),
                constraint_pass_rate=_metadata_float(
                    verifier_metadata,
                    "constraint_pass_rate",
                    fallback_keys=("hard_constraint_pass_rate",),
                ),
            )
        if spec.formula_name == "research_writing_match":
            return compute_research_writing_match_reward(
                grounding_score=float(verifier_metadata.get("grounding_score") or 0.0),
                constraint_compliance=float(verifier_metadata.get("constraint_compliance") or 0.0),
                coverage_score=float(verifier_metadata.get("coverage_score") or 0.0),
            )
        raise ValueError(f"unsupported reward formula: {spec.formula_name}")


def _merge_verifier_metadata(verifier_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for record in verifier_results:
        if "result" in record:
            result = record.get("result") or {}
            metadata = result.get("metadata") or {}
        else:
            metadata = record.get("metadata") or {}
        merged.update(metadata)
    return merged


def _metadata_float(metadata: dict[str, Any], primary_key: str, *, fallback_keys: tuple[str, ...] = ()) -> float:
    for key in (primary_key, *fallback_keys):
        if key not in metadata or metadata[key] is None:
            continue
        try:
            return float(metadata[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def _safe_step_count(raw_steps: Any, *, fallback: int) -> int:
    if raw_steps is None:
        return fallback
    try:
        return int(raw_steps)
    except (TypeError, ValueError):
        return fallback


def _default_reward_spec_for_profile(profile_id: str) -> str:
    # Keep live profile ids mapped to explicit reward specs instead of relying on implicit name matches.
    return {
        "coding-default": "coding_swe_efficiency",
        "researcher": "research_writing_match",
        "text-ui": "ui_from_text_layout",
        "image-ui": "ui_from_text_layout",
    }.get(profile_id, profile_id)


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
    outcome: AdvisorOutcome,
    constraint_violations: list[str] | None,
    diagnostics: dict[str, Any],
) -> str | None:
    if constraint_violations:
        return "constraint_failure"
    if outcome.status == "failure":
        return "failed_execution"
    if diagnostics.get("screenshot_similarity", 1.0) < 0.5:
        return "targeting_miss"
    if diagnostics.get("steps", 0) > 20:
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
