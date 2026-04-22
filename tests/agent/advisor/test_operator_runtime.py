import json
import sqlite3
from datetime import UTC, datetime

from agent.advisor.api import create_orchestrator
from agent.advisor.benchmark import BenchmarkRunManifest
from agent.advisor.operator_runtime import (
    OperatorJobQueue,
    RetentionEnforcer,
    build_deployment_profile,
    build_operator_snapshot,
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
            confidence=0.9,
        )


def _packet(run_id: str):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="repair main flow",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False, "session_id": f"sess-{run_id}"},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests must pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["repair succeeds"],
        token_budget=900,
    )


def _seed_run(tmp_path, run_id: str = "run-operator"):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        event_log_path=str(tmp_path / "events.jsonl"),
        retention_days=30,
    )
    orchestrator = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="success",
                summary="patched main.py",
                output="patched main.py",
                files_touched=["main.py"],
                tests_run=["pytest -q"],
                metadata={"provider": "stub"},
            ),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )
    result = orchestrator.run(_packet(run_id))
    return store, settings, result


def test_build_deployment_profile_and_snapshot_summarize_operator_state(tmp_path):
    store, settings, result = _seed_run(tmp_path)
    deployment = build_deployment_profile(settings=settings, mode="hosted")
    benchmark_summary = [
        BenchmarkRunManifest(
            run_id="baseline-run",
            fixture_id="coding-main",
            domain="coding",
            split="validation",
            packet_hash="abc",
            executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
            verifier_set=["build-check"],
            routing_arm="baseline",
            reward_version="phase8-v1",
            score={"overall_score": 0.5, "focus_target_recall": 0.5},
        ),
        BenchmarkRunManifest(
            run_id="advisor-run",
            fixture_id="coding-main",
            domain="coding",
            split="validation",
            packet_hash="abc",
            executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
            verifier_set=["build-check"],
            routing_arm="advisor",
            reward_version="phase8-v1",
            score={"overall_score": 0.7, "focus_target_recall": 0.8},
        ),
    ]

    snapshot = build_operator_snapshot(
        store=store,
        settings=settings,
        deployment=deployment,
        benchmark_manifests=benchmark_summary,
        job_records=[],
    )

    assert deployment.mode == "hosted"
    assert deployment.bind_host == "0.0.0.0"
    assert snapshot["deployment"]["auth_boundary"] == "external auth proxy required"
    assert snapshot["live_metrics"]["total_runs"] == 1
    assert snapshot["runs"][0]["run_id"] == result.run_id
    assert snapshot["runs"][0]["lineage_available"] is True
    assert snapshot["benchmark_summary"]["deltas"]["advisor_minus_baseline"]["overall_score"] == 0.2


def test_operator_job_queue_persists_and_resumes_incomplete_jobs(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    queued = queue.enqueue_job(job_type="benchmark", payload={"suite": "core"}, resume_token="suite-core")
    running = queue.update_job(queued.job_id, status="running")
    failed = queue.update_job(queued.job_id, status="failed", last_error="timeout")
    resumed = queue.resume_incomplete_jobs()
    persisted = queue.list_jobs()

    assert queued.status == "queued"
    assert running.status == "running"
    assert failed.last_error == "timeout"
    assert [job.job_id for job in resumed] == [queued.job_id]
    assert persisted[0].status == "queued"
    assert persisted[0].resume_token == "suite-core"


def test_retention_enforcer_archives_old_runs_and_rotates_event_logs(tmp_path):
    store, settings, result = _seed_run(tmp_path, run_id="run-old")
    old_ts = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE runs SET started_at = ? WHERE run_id = ?", (old_ts, result.run_id))
        conn.execute("UPDATE run_outcomes SET completed_at = ? WHERE run_id = ?", (old_ts, result.run_id))

    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-01-01T00:00:00+00:00", "event_type": "old", "run_id": result.run_id, "stage": "executor", "payload": {}}),
                json.dumps({"ts": "2026-02-15T00:00:00+00:00", "event_type": "new", "run_id": "run-new", "stage": "executor", "payload": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    enforcer = RetentionEnforcer(store=store, settings=settings, archive_root=tmp_path / "archive")
    report = enforcer.enforce(now=datetime(2026, 2, 15, tzinfo=UTC))

    remaining_events = event_log.read_text(encoding="utf-8").splitlines()

    assert report["archived_runs"] == 1
    assert report["deleted_runs"] == 1
    assert report["archived_event_lines"] == 1
    assert store.get_run(result.run_id) is None
    assert len(remaining_events) == 1
    assert json.loads(remaining_events[0])["event_type"] == "new"
