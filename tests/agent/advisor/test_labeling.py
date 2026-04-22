import json

from agent.advisor.labeling import export_training_examples
from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    CandidateFile,
    RepoSummary,
)
from agent.advisor.trace_store import AdvisorTraceStore


def _packet(run_id: str):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="fix prompt builder",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=[],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=900,
    )


def test_export_training_examples_writes_jsonl(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet("run-1")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=0, tests_run=["pytest -q"], review_verdict="pass")
    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="abc")
    store.record_outcome(outcome)

    out = tmp_path / "train.jsonl"
    count = export_training_examples(store, out)
    assert count == 1
    rows = out.read_text().strip().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["run_id"] == "run-1"
    assert payload["input"]["task_text"] == "fix prompt builder"
    assert payload["target_advice"]["recommended_plan"] == ["inspect main.py"]
