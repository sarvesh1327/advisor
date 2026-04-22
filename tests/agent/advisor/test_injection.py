from agent.advisor.injector import render_advice_for_user_context
from agent.advisor.schemas import AdviceBlock, RelevantFile


def test_render_advice_for_user_context_contains_sections():
    advice = AdviceBlock(
        task_type="bugfix",
        relevant_files=[RelevantFile(path="main.py", why="application entrypoint", priority=1)],
        recommended_plan=["inspect main.py", "run targeted test"],
        confidence=0.8,
    )
    rendered = render_advice_for_user_context(advice)
    assert "Advisor hint" in rendered
    assert "main.py" in rendered
    assert "run targeted test" in rendered


def test_render_advice_for_user_context_includes_optional_sections():
    advice = AdviceBlock(
        task_type="feature",
        constraints=["preserve API shape"],
        likely_failure_modes=["editing an unrelated module"],
        avoid=["broad refactors"],
        confidence=0.6,
    )
    rendered = render_advice_for_user_context(advice)
    assert "Constraints:" in rendered
    assert "Likely failure modes:" in rendered
    assert "Avoid:" in rendered
    assert "Confidence: 0.60" in rendered
