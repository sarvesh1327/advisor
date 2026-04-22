from agent.advisor.metrics import summarize_runs
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, CandidateFile, RelevantFile, RepoSummary
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


def test_summarize_runs_counts_success_and_file_hit_rate(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet("run-1")
    advice = AdviceBlock(task_type="bugfix", relevant_files=[RelevantFile(path="main.py", why="application entrypoint", priority=1)], confidence=0.8)
    outcome = AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=1, tests_run=["pytest -q"])
    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="abc")
    store.record_outcome(outcome)

    summary = summarize_runs(store)
    assert summary["total_runs"] == 1
    assert summary["success_runs"] == 1
    assert summary["file_hit_rate"] == 1.0
