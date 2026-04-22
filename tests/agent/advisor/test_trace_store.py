from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    CandidateFile,
    RepoSummary,
)
from agent.advisor.trace_store import AdvisorTraceStore


def _packet(run_id: str = "run-1"):
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


def test_trace_store_roundtrip(tmp_path):
    db_path = tmp_path / "advisor.db"
    store = AdvisorTraceStore(db_path)
    packet = _packet()
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=1, tests_run=["pytest -q"], review_verdict="pass")

    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="abc")
    store.record_outcome(outcome)

    row = store.get_run("run-1")
    assert row is not None
    assert row["run_id"] == "run-1"
    assert row["advice"]["recommended_plan"] == ["inspect main.py"]
    assert row["outcome"]["status"] == "success"
