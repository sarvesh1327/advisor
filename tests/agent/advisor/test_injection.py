from agent.advisor.core.injector import render_advice_for_user_context
from agent.advisor.core.schemas import AdviceBlock, FocusTarget


def test_render_advice_for_user_context_contains_sections():
    advice = AdviceBlock(
        task_type="bugfix",
        focus_targets=[FocusTarget(kind="file", locator="main.py", rationale="application entrypoint", priority=1)],
        recommended_plan=["inspect main.py", "run targeted test"],
        confidence=0.8,
    )
    rendered = render_advice_for_user_context(advice)
    assert "Advisor hint" in rendered
    assert "Focus targets:" in rendered
    assert "main.py" in rendered
    assert "run targeted test" in rendered


def test_render_advice_for_user_context_includes_optional_sections():
    advice = AdviceBlock(
        task_type="feature",
        focus_targets=[FocusTarget(kind="doc", locator="docs/brief.md", rationale="task brief", priority=1)],
        constraints=["preserve API shape"],
        likely_failure_modes=["editing an unrelated module"],
        avoid=["broad refactors"],
        confidence=0.6,
        notes="Confidence is moderate until the current brief is re-checked.",
    )
    rendered = render_advice_for_user_context(advice)
    assert "Constraints:" in rendered
    assert "Likely failure modes:" in rendered
    assert "Avoid:" in rendered
    assert "Confidence: 0.60" in rendered
    assert "verify before acting on uncertain advice" in rendered
