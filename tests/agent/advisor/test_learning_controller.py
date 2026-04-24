import sqlite3
import textwrap
from pathlib import Path

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.execution.orchestration import DeterministicABRouter, ExecutorRunResult, FrontierChatExecutor
from agent.advisor.learning.controller import AutonomousLearningController
from agent.advisor.learning.readiness import build_learning_readiness_report, collect_fresh_rollout_groups
from agent.advisor.learning.state import AutonomousLearningState
from agent.advisor.product.api import create_orchestrator
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.storage.trace_store import AdvisorTraceStore


class StubRuntime:
    def generate_advice(self, packet, system_prompt=None, advisor_profile_id=None):
        del system_prompt, advisor_profile_id
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect likely file"], confidence=0.8)


ADVISOR_REPO = str(Path("/Users/clawuser/Desktop/Black-box steer/Advisor").expanduser())


def _packet(run_id: str, repo_path: str = ADVISOR_REPO):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="improve advisor autonomous learning",
        task_type="bugfix",
        repo={"path": repo_path, "branch": "main", "dirty": False, "session_id": f"sess-{run_id}"},
        repo_summary=RepoSummary(modules=["agent"], hotspots=["agent/advisor"], file_tree_slice=["agent/advisor"]),
        candidate_files=[CandidateFile(path="agent/advisor/product/hardening.py", reason="target overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests must pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["autonomous learning works"],
        token_budget=900,
    )


def _seed_dogfood_run(tmp_path, run_id: str, profile_id: str = "coding-default"):
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        event_log_path=str(tmp_path / "events.jsonl"),
        advisor_profiles_path=str(Path(ADVISOR_REPO) / "config" / "advisor_profiles.toml"),
    )
    store = AdvisorTraceStore(settings.trace_db_path)
    step_count = 2 if run_id.endswith("1") else 6
    orchestrator = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="success",
                summary="patched advisor file",
                output="patched advisor file",
                files_touched=["agent/advisor/product/hardening.py"],
                tests_run=["pytest -q"],
                metadata={"provider": "stub", "steps": step_count, "render_valid": True},
            ),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )
    result = orchestrator.run(_packet(run_id), advisor_profile_id=profile_id)
    return settings, store, result


def _delete_trajectories(settings: AdvisorSettings) -> None:
    with sqlite3.connect(settings.trace_db_path) as conn:
        conn.execute("DELETE FROM advisor_trajectories")



def _clear_trajectory_rewards(settings: AdvisorSettings) -> None:
    with sqlite3.connect(settings.trace_db_path) as conn:
        conn.execute("UPDATE advisor_trajectories SET final_reward_json = NULL")



def _seed_dogfood_runs(tmp_path, prefix: str, count: int = 4):
    first_settings = None
    first_store = None
    for index in range(1, count + 1):
        settings, store, _ = _seed_dogfood_run(tmp_path, f"{prefix}-{index}")
        first_settings = first_settings or settings
        first_store = first_store or store
    assert first_settings is not None
    assert first_store is not None
    return first_settings, first_store



def _write_zero_group_profiles(tmp_path):
    config_path = tmp_path / "profiles-zero-group.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            default_profile_id = "coding-default"

            [profiles.coding-default]
            domain = "coding"
            description = "Default coding advisor profile"

            [profiles.coding-default.training]
            backend = "grpo"
            rollout_group_size = 0
            num_generations = 8
            max_steps = 12
            max_prompt_tokens = 4096
            max_completion_tokens = 1024
            checkpoint_root = "artifacts/checkpoints/coding-default"
            base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
            adapter_method = "lora"
            lora_rank = 32
            lora_alpha = 64
            lora_dropout = 0.05
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            """
        ).strip()
    )
    return AdvisorProfileRegistry.from_toml(config_path)


def test_autonomous_learning_controller_consumes_real_dogfood_runs_and_persists_state(tmp_path, monkeypatch):
    settings, store = _seed_dogfood_runs(tmp_path, "run-dogfood")

    def fake_cycle_runner(queue, **kwargs):
        del queue
        return {
            "train_job": {"job_id": "job-train", "result": {"checkpoint_id": "ckpt-1"}},
            "eval_job": {"job_id": "job-eval", "result": {"promote": True}},
            "promote_job": {"job_id": "job-promote", "result": {"promoted": True, "status": "active"}},
            "promoted": True,
        }

    monkeypatch.setattr("agent.advisor.learning.controller.run_continuous_training_cycle", fake_cycle_runner)

    controller = AutonomousLearningController(settings=settings, trace_store=store)
    tick = controller.tick()
    _seed_dogfood_runs(tmp_path, "run-dogfood-next")
    second_tick = controller.tick()
    status = controller.controller_status()

    assert tick["launched_profiles"] == ["coding-default"]
    assert tick["cycle_results"]["coding-default"]["promoted"] is True
    assert second_tick["launched_profiles"] == ["coding-default"]
    profile_state = status["profiles"]["coding-default"]
    assert len(profile_state["consumed_trajectory_ids"]) == 8
    assert len(profile_state["consumed_run_ids"]) == 8
    assert profile_state["last_cycle_experiment_id"] is not None
    assert profile_state["last_cycle_completed_at"] is not None
    assert profile_state["active_cycle_job_ids"] == []


def test_autonomous_learning_controller_applies_backoff_and_profile_pause_after_repeated_failures(tmp_path, monkeypatch):
    settings, store = _seed_dogfood_runs(tmp_path, "run-fail")

    def failing_cycle_runner(queue, **kwargs):
        del queue, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.advisor.learning.controller.run_continuous_training_cycle", failing_cycle_runner)

    controller = AutonomousLearningController(settings=settings, trace_store=store)
    state = controller.load_state()
    state.policy.max_consecutive_failures = 2
    state.policy.backoff_seconds = 0
    controller.save_state(state)

    first = controller.tick()
    second = controller.tick()
    status = controller.controller_status()

    assert "coding-default" in first["skipped_profiles"]
    profile_state = status["profiles"]["coding-default"]
    assert profile_state["consecutive_failures"] == 2
    assert profile_state["paused"] is True
    assert profile_state["paused_reason"] == "max_consecutive_failures"
    assert profile_state["backoff_until"] is not None
    assert "coding-default" in second["skipped_profiles"]


def test_autonomous_learning_controller_readiness_report_blocks_when_no_fresh_runs_remain(tmp_path, monkeypatch):
    settings, store = _seed_dogfood_runs(tmp_path, "run-ready")

    monkeypatch.setattr(
        "agent.advisor.learning.controller.run_continuous_training_cycle",
        lambda queue, **kwargs: {
            "train_job": {"job_id": "job-train", "result": {"checkpoint_id": "ckpt-1"}},
            "eval_job": {"job_id": "job-eval", "result": {"promote": False}},
            "promote_job": None,
            "promoted": False,
        },
    )

    controller = AutonomousLearningController(settings=settings, trace_store=store)
    controller.tick()
    readiness = controller.readiness_report("coding-default")

    assert readiness["ready"] is False
    assert "insufficient_fresh_runs" in readiness["blocking_reasons"]


def test_collect_fresh_rollout_groups_blocks_non_positive_rollout_group_size(tmp_path):
    settings, store = _seed_dogfood_runs(tmp_path, "run-zero")
    registry = _write_zero_group_profiles(tmp_path)
    state = AutonomousLearningState()

    report = build_learning_readiness_report(
        store=store,
        registry=registry,
        state=state,
        advisor_profile_id="coding-default",
    )

    assert report.ready is False
    assert "invalid_rollout_group_size" in report.blocking_reasons
    assert (
        collect_fresh_rollout_groups(
            store=store,
            registry=registry,
            state=state,
            advisor_profile_id="coding-default",
        )
        is None
    )



def test_learning_readiness_blocks_when_fresh_trajectories_are_below_rollout_group_size(tmp_path):
    settings, store = _seed_dogfood_runs(tmp_path, "run-under-sized", count=2)

    report = build_learning_readiness_report(
        store=store,
        registry=AdvisorProfileRegistry.from_toml(settings.advisor_profiles_path),
        state=AutonomousLearningState(),
        advisor_profile_id="coding-default",
    )

    assert report.ready is False
    assert report.fresh_trajectory_count == 2
    assert report.summary["required_rollout_group_size"] == 4
    assert "insufficient_fresh_trajectories" in report.blocking_reasons



def test_learning_readiness_blocks_rewarded_runs_without_trajectories(tmp_path):
    settings, store = _seed_dogfood_runs(tmp_path, "run-no-trajectory")
    _delete_trajectories(settings)

    report = build_learning_readiness_report(
        store=store,
        registry=AdvisorProfileRegistry.from_toml(settings.advisor_profiles_path),
        state=AutonomousLearningState(),
        advisor_profile_id="coding-default",
    )

    assert report.ready is False
    assert report.fresh_run_count == 4
    assert report.fresh_trajectory_count == 0
    assert "insufficient_fresh_trajectories" in report.blocking_reasons



def test_learning_readiness_blocks_trajectories_without_final_reward(tmp_path):
    settings, store = _seed_dogfood_runs(tmp_path, "run-unrewarded-trajectory")
    _clear_trajectory_rewards(settings)

    report = build_learning_readiness_report(
        store=store,
        registry=AdvisorProfileRegistry.from_toml(settings.advisor_profiles_path),
        state=AutonomousLearningState(),
        advisor_profile_id="coding-default",
    )

    assert report.ready is False
    assert report.fresh_run_count == 4
    assert report.fresh_trajectory_count == 0
    assert "insufficient_fresh_trajectories" in report.blocking_reasons



def test_collect_fresh_rollout_groups_uses_unconsumed_rewarded_trajectories(tmp_path):
    settings, store = _seed_dogfood_runs(tmp_path, "run-trajectory-source")
    registry = AdvisorProfileRegistry.from_toml(settings.advisor_profiles_path)
    state = AutonomousLearningState()

    collection = collect_fresh_rollout_groups(
        store=store,
        registry=registry,
        state=state,
        advisor_profile_id="coding-default",
    )

    assert collection is not None
    assert len(collection.trajectory_ids) == 4
    assert len(collection.run_ids) == 4
    assert collection.rollout_group.summary["source_trajectory_ids"] == collection.trajectory_ids
    assert all(result.trajectory for result in collection.rollout_group.results)
