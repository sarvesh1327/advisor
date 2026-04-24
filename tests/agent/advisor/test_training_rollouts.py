from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.execution.orchestration import (
    BuildTestVerifier,
    ExecutorRunResult,
    ExecutorStepResult,
    FrontierChatExecutor,
    VerifierResult,
)
from agent.advisor.rewards.reward_registry import RewardRegistry
from agent.advisor.training.training_rollouts import (
    RolloutTurnRecord,
    TrainingRolloutGroupRequest,
    TrainingRolloutRequest,
    execute_training_rollout,
    execute_training_rollout_group,
)


class StubRuntime:
    def __init__(self):
        self.calls = []

    def generate_advice(self, packet, system_prompt=None):
        self.calls.append({"packet": packet, "system_prompt": system_prompt})
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect main.py"], confidence=0.9)


class StepRuntime:
    def __init__(self):
        self.calls = []

    def generate_advice(self, packet, system_prompt=None, advisor_profile_id=None):
        self.calls.append({"packet": packet, "system_prompt": system_prompt, "advisor_profile_id": advisor_profile_id})
        return AdviceBlock(task_type=packet.task_type, recommended_plan=[f"turn {len(self.calls)} plan"], confidence=0.9)


class SequencedStepExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def execute_step(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def _packet(run_id: str, task_type: str = "bugfix"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="repair main flow",
        task_type=task_type,
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="entrypoint", score=0.9)],
        recent_failures=[],
        constraints=["tests must pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["repair succeeds"],
        token_budget=900,
    )


def test_execute_training_rollout_produces_profile_local_single_turn_result():
    runtime = StubRuntime()
    executor = FrontierChatExecutor(
        name="frontier-chat",
        execute_fn=lambda request: ExecutorRunResult(
            status="success",
            summary="patched main.py",
            output="patched main.py",
            files_touched=["main.py"],
            tests_run=["pytest -q"],
            metadata={"steps": 5},
        ),
    )
    verifier = BuildTestVerifier(
        name="build-check",
        verify_fn=lambda request, result: VerifierResult(status="pass", summary="tests green"),
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-1",
        advisor_profile_id="coding-default",
        packet=_packet("run-rollout-1"),
        executor_name="frontier-chat",
        executor_kind="frontier_chat",
        verifier_names=["build-check"],
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[verifier],
        reward_registry=RewardRegistry.default(),
    )

    assert result.rollout_id == "rollout-1"
    assert result.advisor_profile_id == "coding-default"
    assert result.executor_result.status == "success"
    assert result.reward_label.reward_profile_id == "coding_swe_efficiency"
    assert result.reward_label.total_reward == 0.9375
    assert result.diagnostics["verifier_count"] == 1


def test_execute_training_rollout_calls_runtime_once_per_turn_until_success():
    runtime = StepRuntime()
    executor = SequencedStepExecutor(
        [
            ExecutorStepResult(status="partial", summary="draft patch", files_touched=["main.py"], done=False),
            ExecutorStepResult(status="success", summary="tests green", files_touched=["main.py"], tests_run=["pytest -q"], done=True),
        ]
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-multiturn-success",
        advisor_profile_id="coding-default",
        packet=_packet("run-multiturn-success"),
        executor_name="step-executor",
        executor_kind="coding_agent",
        max_turns=3,
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert len(runtime.calls) == 2
    assert [item.turn_index for item in executor.requests] == [0, 1]
    assert result.executor_result.status == "success"
    assert result.diagnostics["stop_reason"] == "success"


def test_execute_training_rollout_threads_previous_observations_into_next_packet():
    runtime = StepRuntime()
    executor = SequencedStepExecutor(
        [
            ExecutorStepResult(status="partial", summary="first observation", error_messages=["lint failed"], done=False),
            ExecutorStepResult(status="success", summary="second observation", done=True),
        ]
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-observations",
        advisor_profile_id="coding-default",
        packet=_packet("run-observations"),
        executor_name="step-executor",
        executor_kind="coding_agent",
        max_turns=2,
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    second_packet = runtime.calls[1]["packet"]
    assert second_packet.history[-1].kind == "turn-observation"
    assert second_packet.history[-1].summary == "first observation"
    assert executor.requests[1].previous_observations[0].error_messages == ["lint failed"]
    assert result.diagnostics["turn_count"] == 2


def test_execute_training_rollout_returns_trajectory_with_final_reward():
    runtime = StepRuntime()
    executor = SequencedStepExecutor(
        [ExecutorStepResult(status="success", summary="done", files_touched=["main.py"], tests_run=["pytest -q"], done=True)]
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-trajectory",
        advisor_profile_id="coding-default",
        packet=_packet("run-trajectory"),
        executor_name="step-executor",
        executor_kind="coding_agent",
        max_turns=2,
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert result.trajectory["trajectory_id"] == "rollout-trajectory"
    assert result.trajectory["turns"][0]["state_packet"]["run_id"] == "run-trajectory"
    assert result.trajectory["turns"][0]["advice"]["recommended_plan"] == ["turn 1 plan"]
    assert result.trajectory["turns"][0]["observation"]["status"] == "success"
    assert result.trajectory["final_reward"]["total_reward"] == result.reward_label.total_reward


def test_execute_training_rollout_respects_max_turns():
    runtime = StepRuntime()
    executor = SequencedStepExecutor(
        [
            ExecutorStepResult(status="partial", summary="still drafting", done=False),
            ExecutorStepResult(status="partial", summary="still blocked", done=False),
            ExecutorStepResult(status="success", summary="should not run", done=True),
        ]
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-max-turns",
        advisor_profile_id="coding-default",
        packet=_packet("run-max-turns"),
        executor_name="step-executor",
        executor_kind="coding_agent",
        max_turns=2,
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert len(runtime.calls) == 2
    assert result.executor_result.status == "partial"
    assert result.diagnostics["stop_reason"] == "max_turns"
    assert len(result.trajectory["turns"]) == 2


def test_execute_training_rollout_preserves_multi_turn_transcript():
    runtime = StubRuntime()
    executor = FrontierChatExecutor(
        name="coding-agent",
        execute_fn=lambda request: ExecutorRunResult(
            status="partial",
            summary="draft patch ready",
            output="draft patch",
            files_touched=["main.py"],
            tests_run=[],
            metadata={"steps": 8},
        ),
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-2",
        advisor_profile_id="coding-default",
        packet=_packet("run-rollout-2", task_type="swe-loop"),
        executor_name="coding-agent",
        executor_kind="coding_agent",
        verifier_names=[],
        max_turns=4,
        multi_turn_transcript=[
            RolloutTurnRecord(turn_index=1, actor="advisor", content="inspect main.py"),
            RolloutTurnRecord(turn_index=2, actor="executor", content="draft patch ready"),
        ],
    )

    result = execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert len(result.multi_turn_transcript) == 2
    assert result.multi_turn_transcript[0].actor == "advisor"
    assert result.diagnostics["multi_turn"] is True


def test_execute_training_rollout_group_summarizes_reward_values():
    runtime = StubRuntime()
    executor = FrontierChatExecutor(
        name="frontier-chat",
        execute_fn=lambda request: ExecutorRunResult(
            status="success",
            summary="patched main.py",
            output="patched main.py",
            files_touched=["main.py"],
            tests_run=["pytest -q"],
            metadata={"steps": 4},
        ),
    )
    group_request = TrainingRolloutGroupRequest(
        group_id="group-1",
        advisor_profile_id="coding-default",
        requests=[
            TrainingRolloutRequest(
                rollout_id="rollout-a",
                advisor_profile_id="coding-default",
                packet=_packet("run-group-a"),
                executor_name="frontier-chat",
                executor_kind="frontier_chat",
                verifier_names=[],
                candidate_index=0,
                group_id="group-1",
            ),
            TrainingRolloutRequest(
                rollout_id="rollout-b",
                advisor_profile_id="coding-default",
                packet=_packet("run-group-b"),
                executor_name="frontier-chat",
                executor_kind="frontier_chat",
                verifier_names=[],
                candidate_index=1,
                group_id="group-1",
            ),
        ],
    )

    group_result = execute_training_rollout_group(
        group_request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert group_result.group_id == "group-1"
    assert group_result.rollout_count == 2
    assert group_result.reward_values == [0.95, 0.95]
    assert group_result.summary["mean_reward"] == 0.95


class ProfileAwareRuntime(StubRuntime):
    def generate_advice(self, packet, system_prompt=None, advisor_profile_id=None):
        self.calls.append(
            {
                "packet": packet,
                "system_prompt": system_prompt,
                "advisor_profile_id": advisor_profile_id,
            }
        )
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect main.py"], confidence=0.9)



def test_execute_training_rollout_passes_profile_id_to_profile_aware_runtime():
    runtime = ProfileAwareRuntime()
    executor = FrontierChatExecutor(
        name="frontier-chat",
        execute_fn=lambda request: ExecutorRunResult(status="success", summary="patched", output="patched"),
    )
    request = TrainingRolloutRequest(
        rollout_id="rollout-profile-aware",
        advisor_profile_id="coding-default",
        packet=_packet("run-profile-aware"),
        executor_name="frontier-chat",
        executor_kind="frontier_chat",
        verifier_names=[],
    )

    execute_training_rollout(
        request,
        runtime=runtime,
        executor=executor,
        verifiers=[],
        reward_registry=RewardRegistry.default(),
    )

    assert runtime.calls[0]["advisor_profile_id"] == "coding-default"
