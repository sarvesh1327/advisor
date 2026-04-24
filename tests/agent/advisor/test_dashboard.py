import json

from agent.advisor.core.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTrajectory,
    AdvisorTrajectoryTurn,
    CandidateFile,
    RepoSummary,
    RewardLabel,
    TurnObservation,
)
from agent.advisor.execution.orchestration import (
    ExecutorDescriptor,
    ExecutorRunResult,
    RoutingDecision,
    RunLineage,
    RunManifest,
)
from agent.advisor.product.dashboard import build_advisor_activity_snapshot, render_advisor_activity_dashboard, simplify_run_title
from agent.advisor.storage.trace_store import AdvisorTraceStore
from agent.advisor.training.training_runtime import CheckpointLifecycleManager, TrainingCheckpointRecord


def _packet(run_id: str = "run-dash") -> AdvisorInputPacket:
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="answer the user's follow-up with context",
        task_type="conversation",
        repo={"path": "/tmp/advisor-repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["notes"], hotspots=["transcripts/chat-001.md"], file_tree_slice=["transcripts/chat-001.md"]),
        candidate_files=[CandidateFile(path="transcripts/chat-001.md", reason="latest conversation transcript", score=0.9)],
        recent_failures=[],
        constraints=["stay grounded in prior turns"],
        tool_limits={},
        acceptance_criteria=["answer is coherent"],
        token_budget=900,
    )


def _seed_activity_run(store: AdvisorTraceStore, *, run_id: str = "run-dash", with_reward: bool = False, with_lineage: bool = False, with_trajectory: bool = False) -> None:
    packet = _packet(run_id)
    advice = AdviceBlock(task_type="conversation", recommended_plan=["read prior turns", "answer directly"], confidence=0.8)
    outcome = AdvisorOutcome(
        run_id=packet.run_id,
        status="success",
        files_touched=[],
        retries=0,
        tests_run=[],
        summary="Produced a coherent answer.",
    )
    reward = RewardLabel(run_id=packet.run_id, advisor_profile_id="generalist", total_reward=0.82, quality_score=0.82)
    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-qwen25-3b-v1",
        advisor_profile_id="generalist",
        caller_id="hermes-main",
        latency_ms=42,
        prompt_hash="hash-1",
        injected_advice=advice,
        injected_rendered_advice="[Advisor hint]\n- read prior turns\n- answer directly",
    )
    store.record_outcome(outcome)
    if with_reward:
        store.record_reward_label(reward)
    if with_lineage:
        manifest = RunManifest(
            run_id=packet.run_id,
            routing_decision=RoutingDecision(arm="advisor", advisor_fraction=1.0, routing_key=packet.run_id, bucket=0.0),
            executor=ExecutorDescriptor(name="frontier-chat", kind="frontier_chat"),
        )
        lineage = RunLineage(
            run_id=packet.run_id,
            packet=packet,
            primary_advice=advice,
            executor_result=ExecutorRunResult(status="success", summary="done"),
            outcome=outcome,
            reward_label=reward,
        )
        store.record_lineage(packet.run_id, manifest, lineage)
    if with_trajectory:
        store.record_trajectory(
            AdvisorTrajectory(
                trajectory_id=f"trajectory:{packet.run_id}",
                run_id=packet.run_id,
                advisor_profile_id="generalist",
                task_text=packet.task_text,
                turns=[
                    AdvisorTrajectoryTurn(
                        turn_index=0,
                        state_packet=packet,
                        advice=advice,
                        observation=TurnObservation(turn_index=0, status="success", summary="done"),
                    )
                ],
                final_outcome=outcome,
                final_reward=reward,
                stop_reason="success",
                budget={"max_turns": 1},
            )
        )



def _seed_active_checkpoint(tmp_path):
    artifacts_root = tmp_path / "artifacts"
    lifecycle_manager = CheckpointLifecycleManager(artifacts_root)
    checkpoint_dir = artifacts_root / "checkpoints" / "coding-default" / "ckpt-active"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / "adapters.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
    training_manifest_path = checkpoint_dir / "training-manifest.json"
    backend_manifest_path = checkpoint_dir / "backend-manifest.json"
    adapter_path.write_bytes(b"adapter")
    adapter_config_path.write_text("{}", encoding="utf-8")
    training_manifest_path.write_text(json.dumps({"job_id": "job-train"}), encoding="utf-8")
    backend_manifest_path.write_text(json.dumps({"backend_name": "grpo"}), encoding="utf-8")
    checkpoint_manifest_path.write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-active",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {
                    "adapter_model": str(adapter_path),
                    "adapter_config": str(adapter_config_path),
                    "training_manifest": str(training_manifest_path),
                    "backend_manifest": str(backend_manifest_path),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lifecycle_manager.register_checkpoint(
        TrainingCheckpointRecord(
            checkpoint_id="ckpt-active",
            experiment_id="exp-active",
            path=str(checkpoint_dir),
            status="active",
            benchmark_summary={"overall_score": 0.88},
            advisor_profile_id="coding-default",
        )
    )
    return lifecycle_manager



def test_activity_snapshot_marks_missing_run_evidence_as_blocked(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    _seed_activity_run(store, run_id="run-missing-evidence")

    snapshot = build_advisor_activity_snapshot(store, limit=5)
    run = snapshot["runs"][0]

    assert snapshot["evidence"]["database_counts"] == {
        "runs": 1,
        "outcomes": 1,
        "reward_labels": 0,
        "lineages": 0,
        "trajectories": 0,
    }
    assert run["evidence_badges"] == {
        "trajectory": "no",
        "reward": "no",
        "lineage": "no",
    }
    assert run["evidence_blocked"] is True
    assert snapshot["evidence"]["blocked"] is True
    assert "missing_reward_labels" in snapshot["evidence"]["blocking_reasons"]



def test_activity_snapshot_reports_active_adapter_artifact_evidence(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    lifecycle_manager = _seed_active_checkpoint(tmp_path)

    snapshot = build_advisor_activity_snapshot(
        store,
        lifecycle_manager=lifecycle_manager,
        required_profiles=["coding-default"],
    )
    profile_evidence = snapshot["evidence"]["profiles"]["coding-default"]

    assert profile_evidence["active_checkpoint_id"] == "ckpt-active"
    assert profile_evidence["adapter_file_exists"] is True
    assert profile_evidence["training_manifest_exists"] is True
    assert profile_evidence["backend_manifest_exists"] is True
    assert profile_evidence["blocked"] is False
    assert snapshot["evidence"]["artifact_counts"]["active_checkpoints"] == 1
    assert snapshot["evidence"]["artifact_counts"]["adapter_files_present"] == 1



def test_build_advisor_activity_snapshot_includes_request_advice_and_outcome(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet()
    advice = AdviceBlock(task_type="conversation", recommended_plan=["read prior turns", "answer directly"], confidence=0.8)
    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-qwen25-3b-v1",
        advisor_profile_id="generalist",
        caller_id="hermes-main",
        latency_ms=42,
        prompt_hash="hash-1",
        injected_advice=advice,
        injected_rendered_advice="[Advisor hint]\n- read prior turns\n- answer directly",
    )
    store.record_outcome(
        AdvisorOutcome(
            run_id=packet.run_id,
            status="success",
            files_touched=[],
            retries=0,
            tests_run=[],
            summary="Produced a coherent answer.",
        )
    )

    snapshot = build_advisor_activity_snapshot(store, limit=5)

    assert snapshot["total_runs"] == 1
    assert snapshot["runs"][0]["advisor_profile_id"] == "generalist"
    assert snapshot["runs"][0]["advisor_used"] is True
    assert snapshot["runs"][0]["profile_badge"] == "Profile: generalist"
    assert snapshot["runs"][0]["caller_id"] == "hermes-main"
    assert snapshot["runs"][0]["caller_badge"] == "Caller: hermes-main"
    assert snapshot["runs"][0]["task_text"] == packet.task_text
    assert snapshot["runs"][0]["title"] == packet.task_text
    assert snapshot["runs"][0]["recommended_plan"] == ["read prior turns", "answer directly"]
    assert "Advisor hint" in snapshot["runs"][0]["injected_rendered_advice"]
    assert snapshot["runs"][0]["outcome"]["status"] == "success"


def test_simplify_run_title_strips_large_context_blocks_and_caps_length():
    noisy_task_text = """I see such big context on dashboard. can we turn titles of runs more simpler?

<memory-context>
very long recalled memory that should not appear in card title
</memory-context>

[Advisor middleware]
Run: run_123
Profile: researcher
Advice:
- review the brief
"""

    title = simplify_run_title(noisy_task_text)

    assert title == "I see such big context on dashboard. can we turn titles of runs more simpler?"
    assert "memory-context" not in title
    assert "Advisor middleware" not in title
    assert len(title) <= 90


def test_simplify_run_title_collapses_system_skill_and_maintenance_prompts():
    assert simplify_run_title('[SYSTEM: The user has invoked the "llm-wiki" skill, indicating they want you to follow its instructions.] --- name: llm-wiki') == "Skill: llm-wiki"
    assert simplify_run_title("Review the conversation above and consider saving or updating a skill if appropriate.\n\nFocus on: reusable workflow") == "Skill maintenance check"
    assert simplify_run_title("[SYSTEM: Background process proc_123 completed (exit code 0).]") == "Background process update"


def test_render_advisor_activity_dashboard_uses_simplified_title_not_raw_task_text():
    long_raw_text = """Fix dashboard titles

<memory-context>
this should stay out of h2
</memory-context>

[Advisor middleware]
Run: run_abc
"""

    html = render_advisor_activity_dashboard(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "total_runs": 1,
            "runs": [
                {
                    "run_id": "run-2",
                    "task_text": long_raw_text,
                    "title": simplify_run_title(long_raw_text),
                    "task_type": "conversation",
                    "advisor_profile_id": "generalist",
                    "advisor_used": True,
                    "profile_badge": "Profile: generalist",
                    "caller_id": "hermes-main",
                    "caller_badge": "Caller: hermes-main",
                    "recommended_plan": [],
                    "injected_rendered_advice": "",
                    "outcome": {},
                    "reward_label": None,
                }
            ],
        }
    )

    assert "<h2>Fix dashboard titles</h2>" in html
    assert "this should stay out of h2" not in html


def test_render_advisor_activity_dashboard_renders_run_cards():
    html = render_advisor_activity_dashboard(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "total_runs": 1,
            "runs": [
                {
                    "run_id": "run-1",
                    "started_at": "2026-04-23T00:00:00+00:00",
                    "task_text": "answer the follow-up",
                    "task_type": "conversation",
                    "advisor_profile_id": "generalist",
                    "advisor_used": True,
                    "profile_badge": "Profile: generalist",
                    "caller_id": "hermes-main",
                    "caller_badge": "Caller: hermes-main",
                    "recommended_plan": ["read prior turns", "answer directly"],
                    "injected_rendered_advice": "[Advisor hint] answer directly",
                    "outcome": {"status": "success", "summary": "done"},
                    "reward_label": None,
                    "evidence_badges": {"trajectory": "no", "reward": "no", "lineage": "no"},
                    "evidence_blocked": True,
                }
            ],
        }
    )

    assert "Advisor activity dashboard" in html
    assert "Advisor used: yes" in html
    assert "trajectory: no" in html
    assert "reward: no" in html
    assert "lineage: no" in html
    assert "evidence: blocked" in html
    assert "Profile: generalist" in html
    assert "Caller: hermes-main" in html
    assert "answer the follow-up" in html
    assert "generalist" in html
    assert "answer directly" in html
