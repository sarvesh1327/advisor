from agent.advisor.orchestration import (
    AdvisorOrchestrator,
    BuildTestVerifier,
    DeterministicABRouter,
    ExecutorRunResult,
    FrontierChatExecutor,
    HumanReviewVerifier,
    RubricVerifier,
    VerifierResult,
)
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.settings import AdvisorSettings
from agent.advisor.trace_store import AdvisorTraceStore


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

    result = orchestrator.run(_packet())

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
