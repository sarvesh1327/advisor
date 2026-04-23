import json
import sqlite3
from datetime import UTC, datetime

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.benchmark import BenchmarkRunManifest
from agent.advisor.execution.orchestration import DeterministicABRouter, ExecutorRunResult, FrontierChatExecutor
from agent.advisor.operators.operator_runtime import (
    EvalProfileJobPayload,
    OperatorJobQueue,
    PromoteCheckpointJobPayload,
    RetentionEnforcer,
    TrainProfileJobPayload,
    build_deployment_profile,
    build_operator_snapshot,
    run_continuous_training_cycle,
    run_operator_job,
)
from agent.advisor.product.api import create_orchestrator
from agent.advisor.storage.trace_store import AdvisorTraceStore
from agent.advisor.training.training_runtime import CheckpointLifecycleManager, TrainingCheckpointRecord


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

    queued = queue.enqueue_job(
        job_type="eval-profile",
        payload=EvalProfileJobPayload(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="ckpt-1",
            benchmark_manifests=[
                BenchmarkRunManifest(
                    run_id="baseline-run",
                    fixture_id="coding-main",
                    domain="coding",
                    split="validation",
                    packet_hash="abc",
                    executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                    verifier_set=["build-check"],
                    routing_arm="baseline",
                    advisor_profile_id="coding-default",
                    reward_version="phase8-v1",
                    score={"overall_score": 0.5, "focus_target_recall": 0.5},
                ).model_dump(),
                BenchmarkRunManifest(
                    run_id="advisor-run",
                    fixture_id="coding-main",
                    domain="coding",
                    split="validation",
                    packet_hash="abd",
                    executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                    verifier_set=["build-check"],
                    routing_arm="advisor",
                    advisor_profile_id="coding-default",
                    reward_version="phase8-v1",
                    score={"overall_score": 0.7, "focus_target_recall": 0.8},
                ).model_dump(),
            ],
        ).model_dump(),
        resume_token="suite-core",
    )
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


def test_operator_job_queue_rejects_unknown_job_types_and_invalid_payloads(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")

    try:
        queue.enqueue_job(job_type="unknown-job", payload={})
    except ValueError as exc:
        assert "unsupported job_type" in str(exc)
    else:
        raise AssertionError("expected unknown job type to raise ValueError")

    try:
        queue.enqueue_job(job_type="train-profile", payload={"advisor_profile_id": "coding-default"})
    except ValueError as exc:
        assert "experiment_id" in str(exc)
    else:
        raise AssertionError("expected invalid train-profile payload to raise ValueError")


def test_run_operator_job_executes_train_and_eval_jobs_with_structured_results(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    train_job = queue.enqueue_job(
        job_type="train-profile",
        payload=TrainProfileJobPayload(
            experiment_id="exp-14",
            advisor_profile_id="coding-default",
            rollout_group={
                "group_id": "group-1",
                "advisor_profile_id": "coding-default",
                "results": [],
                "reward_values": [],
                "summary": {},
            },
        ).model_dump(),
        resume_token="train-coding-default",
    )
    eval_job = queue.enqueue_job(
        job_type="eval-profile",
        payload=EvalProfileJobPayload(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="ckpt-1",
            benchmark_manifests=[
                BenchmarkRunManifest(
                    run_id="baseline-run",
                    fixture_id="coding-main",
                    domain="coding",
                    split="validation",
                    packet_hash="hash-a",
                    executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                    verifier_set=["build-check"],
                    routing_arm="baseline",
                    advisor_profile_id="coding-default",
                    reward_version="phase8-v1",
                    score={"overall_score": 0.5, "focus_target_recall": 0.5},
                ).model_dump(),
                BenchmarkRunManifest(
                    run_id="advisor-run",
                    fixture_id="coding-main",
                    domain="coding",
                    split="validation",
                    packet_hash="hash-b",
                    executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                    verifier_set=["build-check"],
                    routing_arm="advisor",
                    advisor_profile_id="coding-default",
                    reward_version="phase8-v1",
                    score={"overall_score": 0.8, "focus_target_recall": 0.8},
                ).model_dump(),
            ],
            promotion_threshold=0.1,
        ).model_dump(),
    )

    train_record = run_operator_job(
        queue,
        train_job.job_id,
        train_profile_fn=lambda payload: {
            "job_kind": "train-profile",
            "advisor_profile_id": payload.advisor_profile_id,
            "checkpoint_id": "ckpt-1",
        },
    )
    eval_record = run_operator_job(
        queue,
        eval_job.job_id,
        eval_profile_fn=lambda payload: {
            "job_kind": "eval-profile",
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promote": True,
        },
    )

    assert train_record.status == "completed"
    assert train_record.result["checkpoint_id"] == "ckpt-1"
    assert eval_record.status == "completed"
    assert eval_record.result["promote"] is True


def test_run_operator_job_records_failures_and_resume_support(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    job = queue.enqueue_job(
        job_type="train-profile",
        payload=TrainProfileJobPayload(
            experiment_id="exp-14",
            advisor_profile_id="coding-default",
            rollout_group={
                "group_id": "group-1",
                "advisor_profile_id": "coding-default",
                "results": [],
                "reward_values": [],
                "summary": {},
            },
        ).model_dump(),
        resume_token="retry-train",
    )

    try:
        run_operator_job(queue, job.job_id, train_profile_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected failing job execution to raise RuntimeError")

    failed = queue.list_jobs()[0]
    resumed = queue.resume_incomplete_jobs()

    assert failed.status == "failed"
    assert failed.last_error == "boom"
    assert [item.job_id for item in resumed] == [job.job_id]
    assert queue.list_jobs()[0].status == "queued"


def test_run_operator_job_returns_completed_record_without_rerunning(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    job = queue.enqueue_job(
        job_type="promote-checkpoint",
        payload=PromoteCheckpointJobPayload(
            advisor_profile_id="coding-default",
            candidate_checkpoint_id="ckpt-1",
            evaluation={
                "advisor_profile_id": "coding-default",
                "candidate_checkpoint_id": "ckpt-1",
                "promote": True,
            },
        ).model_dump(),
    )
    calls = {"count": 0}

    first = run_operator_job(
        queue,
        job.job_id,
        promote_checkpoint_fn=lambda payload: calls.__setitem__("count", calls["count"] + 1) or {"promoted": True},
    )
    second = run_operator_job(
        queue,
        job.job_id,
        promote_checkpoint_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("should not rerun")),
    )

    assert first.status == "completed"
    assert second.status == "completed"
    assert second.result == first.result
    assert calls["count"] == 1



def test_run_continuous_training_cycle_runs_train_eval_and_promote_in_order(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    calls = []

    result = run_continuous_training_cycle(
        queue,
        experiment_id="exp-loop",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-loop",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=[
            BenchmarkRunManifest(
                run_id="baseline-run",
                fixture_id="coding-main",
                domain="coding",
                split="validation",
                packet_hash="hash-a",
                executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                verifier_set=["build-check"],
                routing_arm="baseline",
                advisor_profile_id="coding-default",
                reward_version="phase8-v1",
                score={"overall_score": 0.5, "focus_target_recall": 0.5},
            ).model_dump(),
            BenchmarkRunManifest(
                run_id="advisor-run",
                fixture_id="coding-main",
                domain="coding",
                split="validation",
                packet_hash="hash-b",
                executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
                verifier_set=["build-check"],
                routing_arm="advisor",
                advisor_profile_id="coding-default",
                reward_version="phase8-v1",
                score={"overall_score": 0.8, "focus_target_recall": 0.8},
            ).model_dump(),
        ],
        train_profile_fn=lambda payload: calls.append(("train", payload.advisor_profile_id)) or {
            "checkpoint_id": "ckpt-loop",
            "advisor_profile_id": payload.advisor_profile_id,
        },
        eval_profile_fn=lambda payload: calls.append(("eval", payload.candidate_checkpoint_id)) or {
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promote": True,
        },
        promote_checkpoint_fn=lambda payload: calls.append(("promote", payload.candidate_checkpoint_id)) or {
            "promoted": True,
            "checkpoint_id": payload.candidate_checkpoint_id,
            "advisor_profile_id": payload.advisor_profile_id,
        },
    )

    assert calls == [("train", "coding-default"), ("eval", "ckpt-loop"), ("promote", "ckpt-loop")]
    assert result["train_job"]["status"] == "completed"
    assert result["eval_job"]["status"] == "completed"
    assert result["promote_job"]["status"] == "completed"
    assert result["promoted"] is True



def test_run_continuous_training_cycle_skips_promotion_when_eval_does_not_pass(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    calls = []

    result = run_continuous_training_cycle(
        queue,
        experiment_id="exp-no-promote",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-no-promote",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=[],
        train_profile_fn=lambda payload: calls.append(("train", payload.advisor_profile_id)) or {
            "checkpoint_id": "ckpt-no-promote",
            "advisor_profile_id": payload.advisor_profile_id,
        },
        eval_profile_fn=lambda payload: calls.append(("eval", payload.candidate_checkpoint_id)) or {
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promote": False,
        },
        promote_checkpoint_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("should not promote")),
    )

    assert calls == [("train", "coding-default"), ("eval", "ckpt-no-promote")]
    assert result["promote_job"] is None
    assert result["promoted"] is False



def test_run_continuous_training_cycle_reuses_completed_jobs_on_repeat(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    calls = {"train": 0, "eval": 0, "promote": 0}

    first = run_continuous_training_cycle(
        queue,
        experiment_id="exp-repeat",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-repeat",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=[],
        train_profile_fn=lambda payload: calls.__setitem__("train", calls["train"] + 1) or {
            "checkpoint_id": "ckpt-repeat",
            "advisor_profile_id": payload.advisor_profile_id,
        },
        eval_profile_fn=lambda payload: calls.__setitem__("eval", calls["eval"] + 1) or {
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promote": True,
        },
        promote_checkpoint_fn=lambda payload: calls.__setitem__("promote", calls["promote"] + 1) or {
            "promoted": True,
            "checkpoint_id": payload.candidate_checkpoint_id,
            "advisor_profile_id": payload.advisor_profile_id,
        },
    )
    second = run_continuous_training_cycle(
        queue,
        experiment_id="exp-repeat",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-repeat",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=[],
        train_profile_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("train should not rerun")),
        eval_profile_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("eval should not rerun")),
        promote_checkpoint_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("promote should not rerun")),
    )

    assert first["train_job"]["job_id"] == second["train_job"]["job_id"]
    assert first["eval_job"]["job_id"] == second["eval_job"]["job_id"]
    assert first["promote_job"]["job_id"] == second["promote_job"]["job_id"]
    assert calls == {"train": 1, "eval": 1, "promote": 1}



def _benchmark_manifest_pair(*, baseline_score: float, advisor_score: float) -> list[dict]:
    return [
        BenchmarkRunManifest(
            run_id=f"baseline-{baseline_score}",
            fixture_id="coding-main",
            domain="coding",
            split="validation",
            packet_hash=f"hash-baseline-{baseline_score}",
            executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
            verifier_set=["build-check"],
            routing_arm="baseline",
            advisor_profile_id="coding-default",
            reward_version="phase8-v1",
            score={"overall_score": baseline_score, "focus_target_recall": baseline_score},
        ).model_dump(),
        BenchmarkRunManifest(
            run_id=f"advisor-{advisor_score}",
            fixture_id="coding-main",
            domain="coding",
            split="validation",
            packet_hash=f"hash-advisor-{advisor_score}",
            executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
            verifier_set=["build-check"],
            routing_arm="advisor",
            advisor_profile_id="coding-default",
            reward_version="phase8-v1",
            score={"overall_score": advisor_score, "focus_target_recall": advisor_score},
        ).model_dump(),
    ]



def test_run_continuous_training_cycle_reruns_eval_when_eval_inputs_change(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    calls = {"eval": 0}

    first = run_continuous_training_cycle(
        queue,
        experiment_id="exp-repeat-eval",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-repeat-eval",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=_benchmark_manifest_pair(baseline_score=0.4, advisor_score=0.7),
        promotion_threshold=0.05,
        train_profile_fn=lambda payload: {
            "checkpoint_id": "ckpt-repeat-eval",
            "advisor_profile_id": payload.advisor_profile_id,
        },
        eval_profile_fn=lambda payload: calls.__setitem__("eval", calls["eval"] + 1) or {
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promotion_threshold": payload.promotion_threshold,
            "benchmark_manifest_count": len(payload.benchmark_manifests),
            "promote": False,
        },
    )
    second = run_continuous_training_cycle(
        queue,
        experiment_id="exp-repeat-eval",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-repeat-eval",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=_benchmark_manifest_pair(baseline_score=0.6, advisor_score=0.7),
        promotion_threshold=0.25,
        train_profile_fn=lambda payload: (_ for _ in ()).throw(RuntimeError("train should not rerun")),
        eval_profile_fn=lambda payload: calls.__setitem__("eval", calls["eval"] + 1) or {
            "advisor_profile_id": payload.advisor_profile_id,
            "candidate_checkpoint_id": payload.candidate_checkpoint_id,
            "promotion_threshold": payload.promotion_threshold,
            "benchmark_manifest_count": len(payload.benchmark_manifests),
            "promote": False,
        },
    )

    assert first["eval_job"]["job_id"] != second["eval_job"]["job_id"]
    assert second["eval_job"]["result"]["promotion_threshold"] == 0.25
    assert calls["eval"] == 2



def test_run_continuous_training_cycle_reports_promotion_when_eval_already_activates_checkpoint(tmp_path):
    queue = OperatorJobQueue(tmp_path / "jobs.json")
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        event_log_path=str(tmp_path / "events.jsonl"),
        retention_days=30,
    )
    lifecycle_manager = CheckpointLifecycleManager(tmp_path / "artifacts")

    def _train(payload):
        checkpoint_path = tmp_path / payload.advisor_profile_id / "ckpt-active"
        lifecycle_manager.register_checkpoint(
            TrainingCheckpointRecord(
                checkpoint_id="ckpt-active",
                experiment_id=payload.experiment_id,
                path=str(checkpoint_path),
                status="candidate",
                advisor_profile_id=payload.advisor_profile_id,
            )
        )
        (checkpoint_path / "checkpoint.json").write_text(
            json.dumps({"artifact_paths": {}, "checkpoint_id": "ckpt-active"}),
            encoding="utf-8",
        )
        return {
            "checkpoint_id": "ckpt-active",
            "advisor_profile_id": payload.advisor_profile_id,
        }

    result = run_continuous_training_cycle(
        queue,
        experiment_id="exp-default-promotion",
        advisor_profile_id="coding-default",
        rollout_group={
            "group_id": "group-default-promotion",
            "advisor_profile_id": "coding-default",
            "results": [],
            "reward_values": [],
            "summary": {},
        },
        benchmark_manifests=_benchmark_manifest_pair(baseline_score=0.4, advisor_score=0.7),
        settings=settings,
        lifecycle_manager=lifecycle_manager,
        train_profile_fn=_train,
    )

    checkpoint = lifecycle_manager.get_checkpoint("ckpt-active")

    assert checkpoint is not None
    assert checkpoint.status == "active"
    assert result["eval_job"]["result"]["promote"] is True
    assert result["promote_job"]["result"]["status"] == "noop"
    assert result["promoted"] is True



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
