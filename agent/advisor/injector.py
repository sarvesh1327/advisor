from __future__ import annotations

from .schemas import AdviceBlock


def render_advice_for_user_context(advice: AdviceBlock) -> str:
    # Render the canonical generic advice block that executors actually receive.
    sections: list[str] = ["[Advisor hint — use as guidance, not authority]"]
    if advice.confidence < advice.injection_policy.min_confidence:
        sections.append(
            f"Confidence: {advice.confidence:.2f} (below injection threshold {advice.injection_policy.min_confidence:.2f})"
        )
        sections.append("Calibration: verify before acting on uncertain advice.")
        return "\n\n".join(sections)
    if advice.focus_targets:
        sections.append(
            "Focus targets:\n"
            + "\n".join(
                f"- [{item.kind}] {item.locator}: {item.rationale}" for item in advice.focus_targets
            )
        )
    if advice.constraints:
        sections.append("Constraints:\n" + "\n".join(f"- {item}" for item in advice.constraints))
    if advice.likely_failure_modes:
        sections.append("Likely failure modes:\n" + "\n".join(f"- {item}" for item in advice.likely_failure_modes))
    if advice.recommended_plan:
        sections.append("Suggested plan:\n" + "\n".join(f"- {item}" for item in advice.recommended_plan))
    if advice.avoid:
        sections.append("Avoid:\n" + "\n".join(f"- {item}" for item in advice.avoid))
    if advice.notes:
        sections.append(f"Notes:\n- {advice.notes}")
    if advice.injection_policy.include_confidence_note:
        sections.append(
            f"Confidence: {advice.confidence:.2f}\nCalibration: verify before acting on uncertain advice."
        )
    else:
        sections.append(f"Confidence: {advice.confidence:.2f}")
    return "\n\n".join(sections)
