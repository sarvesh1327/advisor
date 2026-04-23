from __future__ import annotations

from agent.advisor.evaluation.eval_fixtures import EvalFixture
from agent.advisor.evaluation.eval_scoring import score_advice_against_fixture
from agent.advisor.storage.trace_store import AdvisorTraceStore


# Replay should prefer canonical injected advice when available because that is what the executor saw.
def list_replay_runs(store: AdvisorTraceStore) -> list[dict]:
    return store.list_runs(include_context=True)


def evaluate_replay_run(store: AdvisorTraceStore, run_id: str, fixture: EvalFixture) -> dict:
    row = store.get_run(run_id)
    if row is None:
        raise ValueError(f"run not found: {run_id}")
    advice_payload = row.get("injected_advice") or row.get("advice") or {}
    score = score_advice_against_fixture(advice_payload, fixture)
    return {
        "run_id": run_id,
        "fixture_id": fixture.fixture_id,
        "input": row.get("input") or {},
        "advice": advice_payload,
        "score": score,
    }
