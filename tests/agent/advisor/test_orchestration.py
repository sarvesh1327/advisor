from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary, TurnObservation
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.execution.orchestration import (
    AdvisorOrchestrator,
    BuildTestVerifier,
    DeterministicABRouter,
    ExecutorRunResult,
    ExecutorStepRequest,
    ExecutorStepResult,
    FrontierChatExecutor,
    HumanReviewVerifier,
    RubricVerifier,
    VerifierResult,
    run_executor_step,
)
from agent.advisor.storage.trace_store import AdvisorTraceStore


class StubRuntime:
    def __init__(self, advice_blocks):
        self._advice_blocks = list(advice_blocks)
        self.calls = []

    def generate_advice(self, packet, system_prompt=None):
        self.calls.append({"packet": packet, "system_prompt": system_prompt})
        return self._advice_blocks.pop(0)


def _packet(run_id: str = "run-phase10"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="repair the execution loop",
        task_type="bugfix",
        repo={"path": "/tmp/advisor-repo", "branch": "main", "dirty": False, "session_id": "sess-1"},
        repo_summary=RepoSummary(modules=["agent"], hotspots=["gateway.py"], file_tree_slice=["agent/advisor/gateway.py"]),
        candidate_files=[CandidateFile(path="agent/advisor/gateway.py", reason="execution loop lives here", score=0.97)],
        recent_failures=[],
        constraints=["tests must pass", "keep packet generic"],
        tool_limits={"terminal": True},
        acceptance_criteria=["runner persists a replayable lineage"],
        token_budget=900,
    )


def test_run_executor_step_calls_step_capable_executor_once():
    class StepCapableExecutor:
        def __init__(self):
            self.calls = []

        def execute_step(self, request):
            self.calls.append(request)
            return ExecutorStepResult(
                status="partial",
                summary="advanced one step",
                output="draft patch",
                files_touched=["agent/advisor/gateway.py"],
                tests_run=["pytest -q"],
                error_messages=["needs verifier pass"],
                metrics={"tokens": 11},
                done=False,
            )

    executor = StepCapableExecutor()
    request = ExecutorStepRequest(
        trajectory_id="traj-step",
        turn_index=1,
        packet=_packet("run-step"),
        advice=AdviceBlock(task_type="bugfix", recommended_plan=["patch one step"]),
        previous_observations=[TurnObservation(turn_index=0, status="partial", summary="first turn")],
        budget={"max_turns": 3},
    )

    result = run_executor_step(executor, request)

    assert len(executor.calls) == 1
    assert executor.calls[0].trajectory_id == "traj-step"
    assert executor.calls[0].previous_observations[0].summary == "first turn"
    assert result.done is False
    assert result.error_messages == ["needs verifier pass"]


def test_run_executor_step_adapts_legacy_executor_to_terminal_step():
    captured = {}

    def execute(request):
        captured["run_id"] = request.run_id
        captured["rendered_advice"] = request.rendered_advice
        captured["routing_arm"] = request.routing_decision.arm
        return ExecutorRunResult(
            status="success",
            summary="legacy executor completed",
            output="patched gateway.py",
            files_touched=["agent/advisor/gateway.py"],
            tests_run=["pytest -q"],
            metadata={"steps": 1},
            retries=1,
        )

    executor = FrontierChatExecutor(name="legacy-frontier", execute_fn=execute)
    request = ExecutorStepRequest(
        trajectory_id="traj-legacy",
        turn_index=0,
        packet=_packet("run-legacy-step"),
        advice=AdviceBlock(task_type="bugfix", recommended_plan=["run legacy executor"]),
        rendered_advice="advisor hint",
        previous_observations=[],
        budget={"max_turns": 1},
    )

    result = run_executor_step(executor, request)

    assert captured == {"run_id": "run-legacy-step", "rendered_advice": "advisor hint", "routing_arm": "advisor"}
    assert result.status == "success"
    assert result.done is True
    assert result.metrics["steps"] == 1
    assert result.metrics["retries"] == 1


def test_run_executor_step_adapts_step_executor_returning_legacy_result_to_terminal_step():
    class StepExecutorReturningLegacyResult:
        def execute_step(self, request):
            return ExecutorRunResult(
                status="success",
                summary="step reused legacy executor path",
                output="patched gateway.py",
                files_touched=["agent/advisor/gateway.py"],
                tests_run=["pytest -q"],
                metadata={"steps": 2},
                retries=1,
            )

    result = run_executor_step(
        StepExecutorReturningLegacyResult(),
        ExecutorStepRequest(
            trajectory_id="traj-step-legacy",
            turn_index=0,
            packet=_packet("run-step-legacy"),
            advice=AdviceBlock(task_type="bugfix", recommended_plan=["reuse legacy path"]),
        ),
    )

    assert result.done is True
    assert result.metrics["steps"] == 2
    assert result.metrics["retries"] == 1


def test_orchestrator_records_lineage_manifest_and_reward_for_advisor_arm(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = StubRuntime(
        [
            AdviceBlock(
                task_type="bugfix",
                relevant_files=[{"path": "agent/advisor/gateway.py", "why": "execution loop lives here", "priority": 1}],
                recommended_plan=["inspect gateway.py", "run targeted tests"],
                confidence=0.91,
            )
        ]
    )
    executor = FrontierChatExecutor(
        name="frontier-chat",
        execute_fn=lambda request: ExecutorRunResult(
            status="success",
            summary="Executor completed the repair.",
            output="patched gateway.py and ran pytest -q",
            files_touched=["agent/advisor/gateway.py"],
            tests_run=["pytest tests/agent/advisor/test_orchestration.py -q"],
            metadata={"model": "gpt-4.1"},
        ),
    )
    verifiers = [
        BuildTestVerifier(
            name="build-and-test",
            verify_fn=lambda request, result: VerifierResult(status="pass", summary="tests green"),
        ),
        RubricVerifier(
            name="rubric",
            verify_fn=lambda request, result: VerifierResult(status="pass", summary="matches AC"),
        ),
    ]
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=executor,
        verifiers=verifiers,
        router=DeterministicABRouter(advisor_fraction=1.0),
    )

    result = orchestrator.run(_packet(), advisor_profile_id="coding-default")

    assert result.manifest.routing_decision.arm == "advisor"
    assert result.manifest.executor.kind == "frontier_chat"
    assert [item.kind for item in result.manifest.verifiers] == ["build_test", "rubric"]
    assert result.lineage.executor_result.status == "success"
    assert result.lineage.reward_label.example_type == "positive"
    assert result.lineage.reward_label.advisor_profile_id == "coding-default"
    assert result.lineage.reward_label.reward_profile_id == "coding_swe_efficiency"
    assert result.lineage.reward_label.reward_formula == "coding_swe_efficiency"
    assert result.lineage.reward_label.components is None

    persisted = store.get_lineage(result.run_id)
    assert persisted is not None
    assert persisted["manifest"]["routing_decision"]["arm"] == "advisor"
    assert persisted["manifest"]["executor"]["name"] == "frontier-chat"
    assert persisted["lineage"]["reward_label"]["run_id"] == result.run_id


def test_orchestrator_persists_profile_local_trajectory_for_live_run(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = StubRuntime(
        [
            AdviceBlock(
                task_type="bugfix",
                recommended_plan=["inspect gateway.py", "run targeted tests"],
                confidence=0.91,
            )
        ]
    )
    executor = FrontierChatExecutor(
        name="frontier-chat",
        execute_fn=lambda request: ExecutorRunResult(
            status="success",
            summary="Executor completed the repair.",
            output="patched gateway.py and ran pytest -q",
            files_touched=["agent/advisor/gateway.py"],
            tests_run=["pytest tests/agent/advisor/test_orchestration.py -q"],
            metadata={"steps": 3, "model": "gpt-4.1"},
            retries=2,
        ),
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=executor,
        verifiers=[
            BuildTestVerifier(
                name="build-and-test",
                verify_fn=lambda request, result: VerifierResult(
                    status="pass",
                    summary="tests green",
                    metadata={"coverage": "targeted"},
                ),
            )
        ],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )

    result = orchestrator.run(_packet("run-trajectory"), advisor_profile_id="coding-default")

    trajectories = store.list_trajectories(run_id=result.run_id)
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory["trajectory_id"] == f"trajectory:{result.run_id}"
    assert trajectory["run_id"] == result.run_id
    assert trajectory["advisor_profile_id"] == "coding-default"
    assert trajectory["task_text"] == "repair the execution loop"
    assert trajectory["stop_reason"] == "success"
    assert trajectory["budget"] == {"max_turns": 1, "source": "orchestrator.run"}
    assert trajectory["final_outcome"]["status"] == "success"
    assert trajectory["final_reward"]["run_id"] == result.run_id
    assert trajectory["final_reward"]["advisor_profile_id"] == "coding-default"

    assert len(trajectory["turns"]) == 1
    turn = trajectory["turns"][0]
    assert turn["turn_index"] == 0
    assert turn["state_packet"]["run_id"] == result.run_id
    assert turn["advice"]["recommended_plan"] == ["inspect gateway.py", "run targeted tests"]
    assert turn["reward_hint"] == result.lineage.reward_label.total_reward
    assert turn["observation"]["status"] == "success"
    assert turn["observation"]["summary"] == "Executor completed the repair."
    assert turn["observation"]["executor_output"] == "patched gateway.py and ran pytest -q"
    assert turn["observation"]["files_touched"] == ["agent/advisor/gateway.py"]
    assert turn["observation"]["tests_run"] == ["pytest tests/agent/advisor/test_orchestration.py -q"]
    assert turn["observation"]["verifier_hints"] == ["build-and-test: pass — tests green"]
    assert turn["observation"]["metrics"]["steps"] == 3
    assert turn["observation"]["metrics"]["retries"] == 2


def test_orchestrator_uses_profile_reward_registry_not_reward_weight_presets(tmp_path):
    def build_orchestrator(db_name: str, reward_preset: str):
        return AdvisorOrchestrator(
            runtime=StubRuntime([AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.9)]),
            trace_store=AdvisorTraceStore(tmp_path / db_name),
            executor=FrontierChatExecutor(
                name="frontier-chat",
                execute_fn=lambda request: ExecutorRunResult(
                    status="success",
                    summary="Executor completed the repair.",
                    output="patched gateway.py and ran pytest -q",
                    files_touched=["agent/advisor/gateway.py"],
                    tests_run=["pytest tests/agent/advisor/test_orchestration.py -q"],
                    metadata={"steps": 10},
                ),
            ),
            verifiers=[
                BuildTestVerifier(
                    name="build-and-test",
                    verify_fn=lambda request, result: VerifierResult(status="pass", summary="tests green"),
                )
            ],
            router=DeterministicABRouter(advisor_fraction=1.0),
            settings=AdvisorSettings(reward_preset=reward_preset, advisor_profile_id="coding-default"),
        )

    balanced = build_orchestrator("balanced.db", "balanced")
    human_first = build_orchestrator("human-first.db", "human-first")

    balanced_result = balanced.run(_packet("run-balanced"))
    human_first_result = human_first.run(_packet("run-human-first"))

    assert balanced_result.lineage.reward_label.total_reward == 0.875
    assert human_first_result.lineage.reward_label.total_reward == 0.875


def test_orchestrator_routes_baseline_runs_without_injecting_advice(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = StubRuntime([AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.8)])
    captured = {}

    def execute(request):
        captured["rendered_advice"] = request.rendered_advice
        captured["arm"] = request.routing_decision.arm
        return ExecutorRunResult(
            status="success",
            summary="Baseline executor finished.",
            output="completed without advisor injection",
            files_touched=[],
            tests_run=[],
        )

    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(name="frontier-chat", execute_fn=execute),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=0.0),
    )

    result = orchestrator.run(_packet("run-baseline"))

    assert result.manifest.routing_decision.arm == "baseline"
    assert captured == {"rendered_advice": None, "arm": "baseline"}
    assert store.list_trajectories(run_id=result.run_id) == []


def test_orchestrator_supports_optional_second_pass_review(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = StubRuntime(
        [
            AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.82),
            AdviceBlock(task_type="review", recommended_plan=["double-check reward capture"], confidence=0.73),
        ]
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="partial",
                summary="Executor produced a draft patch.",
                output="draft patch ready",
                files_touched=["agent/advisor/gateway.py"],
                tests_run=[],
            ),
        ),
        verifiers=[
            HumanReviewVerifier(
                name="human-review",
                verify_fn=lambda request, result: VerifierResult(status="warn", summary="needs one more look"),
            )
        ],
        router=DeterministicABRouter(advisor_fraction=1.0),
        enable_second_pass_review=True,
    )

    result = orchestrator.run(_packet("run-review"))

    assert len(runtime.calls) == 2
    assert result.lineage.review_advice is not None
    assert result.lineage.review_advice.recommended_plan == ["double-check reward capture"]
    assert result.manifest.review_enabled is True


class ProfileAwareRuntime(StubRuntime):
    def generate_advice(self, packet, system_prompt=None, advisor_profile_id=None):
        self.calls.append(
            {
                "packet": packet,
                "system_prompt": system_prompt,
                "advisor_profile_id": advisor_profile_id,
            }
        )
        return self._advice_blocks.pop(0)



def test_orchestrator_passes_profile_id_to_profile_aware_runtime(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = ProfileAwareRuntime(
        [AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.91)]
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(status="success", summary="done", output="done"),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
        settings=AdvisorSettings(advisor_profile_id="image-ui"),
    )

    orchestrator.run(_packet("run-profile-aware"))

    assert runtime.calls[0]["advisor_profile_id"] == "image-ui"



def test_orchestrator_allows_explicit_profile_override_per_run(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = ProfileAwareRuntime(
        [AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.91)]
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(status="success", summary="done", output="done"),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
        settings=AdvisorSettings(advisor_profile_id="coding-default"),
    )

    result = orchestrator.run(_packet("run-profile-override"), advisor_profile_id="text-ui")
    persisted_run = store.get_run(result.run_id)
    persisted_lineage = store.get_lineage(result.run_id)

    assert runtime.calls[0]["advisor_profile_id"] == "text-ui"
    assert result.lineage.reward_label.advisor_profile_id == "text-ui"
    assert persisted_run is not None
    assert persisted_run["advisor_profile_id"] == "text-ui"
    assert persisted_lineage is not None
    assert persisted_lineage["lineage"]["reward_label"]["advisor_profile_id"] == "text-ui"



def test_orchestrator_review_path_preserves_profile_id_for_profile_aware_runtime(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = ProfileAwareRuntime(
        [
            AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.82),
            AdviceBlock(task_type="review", recommended_plan=["double-check reward capture"], confidence=0.73),
        ]
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(status="partial", summary="draft", output="draft"),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
        enable_second_pass_review=True,
        settings=AdvisorSettings(advisor_profile_id="image-ui"),
    )

    orchestrator.run(_packet("run-profile-review"))

    assert [call["advisor_profile_id"] for call in runtime.calls] == ["image-ui", "image-ui"]



def test_orchestrator_review_path_preserves_explicit_profile_override(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    runtime = ProfileAwareRuntime(
        [
            AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.82),
            AdviceBlock(task_type="review", recommended_plan=["double-check reward capture"], confidence=0.73),
        ]
    )
    orchestrator = AdvisorOrchestrator(
        runtime=runtime,
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(status="partial", summary="draft", output="draft"),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
        enable_second_pass_review=True,
        settings=AdvisorSettings(advisor_profile_id="coding-default"),
    )

    result = orchestrator.run(_packet("run-profile-review-override"), advisor_profile_id="researcher")

    assert [call["advisor_profile_id"] for call in runtime.calls] == ["researcher", "researcher"]
    assert result.lineage.reward_label.advisor_profile_id == "researcher"
