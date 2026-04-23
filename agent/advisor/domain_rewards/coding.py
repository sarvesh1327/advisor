from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def compute_coding_swe_efficiency_reward(*, resolved: bool, steps: int, max_steps: int = 40) -> tuple[float, dict[str, int | bool]]:
    normalized_steps = max(0, min(int(steps), max_steps))
    if not resolved:
        return 0.0, {"resolved": False, "steps": normalized_steps, "max_steps": max_steps}
    reward = 0.5 + 0.5 * ((max_steps - normalized_steps) / max_steps)
    return _clamp(reward), {"resolved": True, "steps": normalized_steps, "max_steps": max_steps}


def compute_coding_exact_answer_reward(*, exact_correct: bool) -> tuple[float, dict[str, bool]]:
    return (1.0 if exact_correct else 0.0), {"exact_correct": exact_correct}
