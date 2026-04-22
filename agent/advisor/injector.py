from __future__ import annotations

from .schemas import AdviceBlock


def render_advice_for_user_context(advice: AdviceBlock) -> str:
    # Render a compact text block that can be prepended to an executor prompt.
    sections: list[str] = ["[Advisor hint — use as guidance, not authority]"]
    if advice.relevant_files:
        sections.append("Relevant files:\n" + "\n".join(f"- {item.path}: {item.why}" for item in advice.relevant_files))
    if advice.constraints:
        sections.append("Constraints:\n" + "\n".join(f"- {item}" for item in advice.constraints))
    if advice.likely_failure_modes:
        sections.append("Likely failure modes:\n" + "\n".join(f"- {item}" for item in advice.likely_failure_modes))
    if advice.recommended_plan:
        sections.append("Suggested plan:\n" + "\n".join(f"- {item}" for item in advice.recommended_plan))
    if advice.avoid:
        sections.append("Avoid:\n" + "\n".join(f"- {item}" for item in advice.avoid))
    sections.append(f"Confidence: {advice.confidence:.2f}")
    return "\n\n".join(sections)
