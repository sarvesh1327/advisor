from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agent.advisor.core.schemas import AdvisorInputPacket


class EvalExpectation(BaseModel):
    # Keep fixture expectations generic-first so the same schema works across domains.
    focus_targets: list[str] = []
    anti_targets: list[str] = []
    required_plan_steps: list[str] = []
    forbidden_plan_steps: list[str] = []
    expected_failure_modes: list[str] = []


class HumanReviewRubric(BaseModel):
    # Human review stays lightweight: fixed scale plus named criteria.
    scale: list[int]
    criteria: list[str]


class EvalFixture(BaseModel):
    fixture_id: str
    domain: str
    description: str
    input_packet: AdvisorInputPacket
    expected_advice: EvalExpectation
    human_review_rubric: HumanReviewRubric


def load_eval_fixture(path: str | Path) -> EvalFixture:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return EvalFixture.model_validate(payload)
