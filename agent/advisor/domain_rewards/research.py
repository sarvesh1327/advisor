from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def compute_research_writing_match_reward(
    *,
    grounding_score: float,
    constraint_compliance: float,
    coverage_score: float,
) -> tuple[float, dict[str, float]]:
    diagnostics = {
        "grounding_score": _clamp(grounding_score),
        "constraint_compliance": _clamp(constraint_compliance),
        "coverage_score": _clamp(coverage_score),
    }
    reward = (diagnostics["grounding_score"] + diagnostics["constraint_compliance"] + diagnostics["coverage_score"]) / 3.0
    return _clamp(reward), diagnostics
