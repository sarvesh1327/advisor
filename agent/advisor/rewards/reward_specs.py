from __future__ import annotations

from pydantic import BaseModel, Field


class RewardSpec(BaseModel):
    reward_spec_id: str
    domain: str
    formula_name: str
    reward_version: str
    required_executor_fields: list[str] = Field(default_factory=list)
    required_verifier_fields: list[str] = Field(default_factory=list)
    diagnostic_fields: list[str] = Field(default_factory=list)


def default_reward_specs() -> dict[str, RewardSpec]:
    return {
        "coding_swe_efficiency": RewardSpec(
            reward_spec_id="coding_swe_efficiency",
            domain="coding",
            formula_name="coding_swe_efficiency",
            reward_version="coding-swe-efficiency-v1",
            diagnostic_fields=["resolved", "steps", "max_steps"],
        ),
        "coding_exact_answer": RewardSpec(
            reward_spec_id="coding_exact_answer",
            domain="coding",
            formula_name="coding_exact_answer",
            reward_version="coding-exact-answer-v1",
            diagnostic_fields=["exact_correct"],
        ),
        "ui_from_text_layout": RewardSpec(
            reward_spec_id="ui_from_text_layout",
            domain="image-ui",
            formula_name="ui_from_text_layout",
            reward_version="ui-from-text-layout-v1",
            diagnostic_fields=["render_valid", "hard_constraint_pass_rate", "soft_style_score"],
        ),
        "ui_edit_from_screenshot": RewardSpec(
            reward_spec_id="ui_edit_from_screenshot",
            domain="image-ui",
            formula_name="ui_edit_from_screenshot",
            reward_version="ui-edit-from-screenshot-v1",
            diagnostic_fields=["render_valid", "screenshot_similarity", "constraint_pass_rate"],
        ),
        "research_writing_match": RewardSpec(
            reward_spec_id="research_writing_match",
            domain="research-writing",
            formula_name="research_writing_match",
            reward_version="research-writing-match-v1",
            diagnostic_fields=["grounding_score", "constraint_compliance", "coverage_score"],
        ),
    }
