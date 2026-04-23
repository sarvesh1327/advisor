from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def compute_ui_from_text_layout_reward(
    *,
    render_valid: bool,
    hard_constraint_pass_rate: float,
    soft_style_score: float,
) -> tuple[float, dict[str, float | bool]]:
    diagnostics = {
        "render_valid": render_valid,
        "hard_constraint_pass_rate": _clamp(hard_constraint_pass_rate),
        "soft_style_score": _clamp(soft_style_score),
    }
    if not render_valid:
        return 0.0, diagnostics
    reward = 0.75 * diagnostics["hard_constraint_pass_rate"] + 0.25 * diagnostics["soft_style_score"]
    return _clamp(reward), diagnostics


def compute_ui_edit_from_screenshot_reward(
    *,
    render_valid: bool,
    screenshot_similarity: float,
    constraint_pass_rate: float,
) -> tuple[float, dict[str, float | bool]]:
    diagnostics = {
        "render_valid": render_valid,
        "screenshot_similarity": _clamp(screenshot_similarity),
        "constraint_pass_rate": _clamp(constraint_pass_rate),
    }
    if not render_valid:
        return 0.0, diagnostics
    reward = 0.7 * diagnostics["screenshot_similarity"] + 0.3 * diagnostics["constraint_pass_rate"]
    return _clamp(reward), diagnostics
