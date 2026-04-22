from agent.advisor.api import create_orchestrator
from agent.advisor.observability import (
    LiveMetricsSnapshot,
    RunEventLogger,
    build_audit_report,
    export_live_metrics,
    redact_packet,
)
from agent.advisor.orchestration import DeterministicABRouter, ExecutorRunResult, FrontierChatExecutor
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.settings import AdvisorSettings
from agent.advisor.trace_store import AdvisorTraceStore


class StubRuntime:
    def generate_advice(self, packet, system_prompt=None):
        return AdviceBlock(
            task_type=packet.task_type,
            relevant_files=[{"path": "main.py", "why": "entrypoint", "priority": 1}],
            recommended_plan=["inspect main.py"],
            confidence=0.88,
        )


def _packet(run_id: str = "run-observe"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="fix login for sarvesh@example.com using secret token abc123",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False, "session_id": "sess-42"},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["do not leak emails or tokens"],
        tool_limits={"terminal": True},
        acceptance_criteria=["fix login"],
        token_budget=900,
    )


def test_redact_packet_masks_sensitive_text_and_respects_safe_defaults():
    packet = _packet()

    redacted = redact_packet(packet)

    assert "sarvesh@example.com" not in redacted["task_text"]
    assert "abc123" not in redacted["task_text"]
    assert "[REDACTED:email]" in redacted["task_text"]
    assert redacted["repo"]["session_id"] == "[REDACTED:id]"


def test_event_logger_writes_structured_run_events(tmp_path):
    logger = RunEventLogger(tmp_path / "events.jsonl")

    logger.log("executor.completed", run_id="run-1", stage="executor", payload={"status": "success"})

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"event_type": "executor.completed"' in lines[0]
    assert '"run_id": "run-1"' in lines[0]


def test_export_live_metrics_and_audit_report_cover_lineage_and_retention(tmp_path):
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        retention_days=30,
        event_log_path=str(tmp_path / "events.jsonl"),
    )
    store = AdvisorTraceStore(settings.trace_db_path)
    orchestrator = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="success",
                summary="done",
                output="fixed main.py",
                files_touched=["main.py"],
                tests_run=["pytest -q"],
            ),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
        enable_second_pass_review=False,
    )

    orchestrator.run(_packet("run-live"))

    snapshot = export_live_metrics(store)
    audit = build_audit_report(store, settings)

    assert isinstance(snapshot, LiveMetricsSnapshot)
    assert snapshot.total_runs == 1
    assert snapshot.lineage_runs == 1
    assert snapshot.reward_labeled_runs == 1
    assert snapshot.arm_counts["advisor"] == 1
    assert audit["retention"]["retention_days"] == 30
    assert audit["dataset_provenance"]["reward_labeled_runs"] == 1
    assert audit["lineage"]["lineage_runs"] == 1
