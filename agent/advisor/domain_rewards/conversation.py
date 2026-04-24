from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def compute_generalist_multi_turn_reward(
    *,
    helpfulness_score: float,
    coherence_score: float,
    constraint_compliance: float,
    grounding_score: float,
) -> tuple[float, dict[str, float]]:
    diagnostics = {
        "helpfulness_score": _clamp(helpfulness_score),
        "coherence_score": _clamp(coherence_score),
        "constraint_compliance": _clamp(constraint_compliance),
        "grounding_score": _clamp(grounding_score),
    }
    reward = (
        0.35 * diagnostics["helpfulness_score"]
        + 0.25 * diagnostics["coherence_score"]
        + 0.25 * diagnostics["constraint_compliance"]
        + 0.15 * diagnostics["grounding_score"]
    )
    return _clamp(reward), diagnostics
