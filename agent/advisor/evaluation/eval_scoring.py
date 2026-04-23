from __future__ import annotations

from agent.advisor.core.schemas import AdviceBlock
from agent.advisor.evaluation.eval_fixtures import EvalFixture


def _normalize_text_set(items: list[str]) -> set[str]:
    return {item.strip().lower() for item in items if item and item.strip()}


# Scores advice using generic targets/plans/failure modes instead of coding-only file assumptions.
def score_advice_against_fixture(advice: AdviceBlock | dict, fixture: EvalFixture) -> dict:
    advice = AdviceBlock.model_validate(advice)
    expected_focus = _normalize_text_set(fixture.expected_advice.focus_targets)
    actual_focus = _normalize_text_set([item.locator for item in advice.focus_targets])
    anti_targets = _normalize_text_set(fixture.expected_advice.anti_targets)
    plan = _normalize_text_set(advice.recommended_plan)
    required_plan = _normalize_text_set(fixture.expected_advice.required_plan_steps)
    forbidden_plan = _normalize_text_set(fixture.expected_advice.forbidden_plan_steps)
    failure_modes = _normalize_text_set(advice.likely_failure_modes)
    expected_failure_modes = _normalize_text_set(fixture.expected_advice.expected_failure_modes)

    focus_hits = actual_focus & expected_focus
    anti_hits = actual_focus & anti_targets
    required_hits = plan & required_plan
    forbidden_hits = plan & forbidden_plan
    failure_hits = failure_modes & expected_failure_modes

    focus_precision = len(focus_hits) / len(actual_focus) if actual_focus else 0.0
    focus_recall = len(focus_hits) / len(expected_focus) if expected_focus else 1.0
    anti_target_violation_rate = len(anti_hits) / len(anti_targets) if anti_targets else 0.0
    required_plan_coverage = len(required_hits) / len(required_plan) if required_plan else 1.0
    forbidden_plan_violation_rate = len(forbidden_hits) / len(forbidden_plan) if forbidden_plan else 0.0
    failure_mode_coverage = len(failure_hits) / len(expected_failure_modes) if expected_failure_modes else 1.0

    overall_score = (
        focus_precision
        + focus_recall
        + (1.0 - anti_target_violation_rate)
        + required_plan_coverage
        + (1.0 - forbidden_plan_violation_rate)
        + failure_mode_coverage
    ) / 6.0

    return {
        "focus_target_precision": focus_precision,
        "focus_target_recall": focus_recall,
        "anti_target_violation_rate": anti_target_violation_rate,
        "required_plan_coverage": required_plan_coverage,
        "forbidden_plan_violation_rate": forbidden_plan_violation_rate,
        "failure_mode_coverage": failure_mode_coverage,
        "overall_score": overall_score,
    }
