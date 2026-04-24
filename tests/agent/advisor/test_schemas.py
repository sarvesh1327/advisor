
from agent.advisor.core.schemas import (
    AdviceBlock,
    AdvisorArtifact,
    AdvisorCapabilityDescriptor,
    AdvisorContext,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTask,
    AdvisorTrajectory,
    AdvisorTrajectoryTurn,
    CandidateFile,
    ExecutorInjectionPolicy,
    FailureSignal,
    FocusTarget,
    RepoSummary,
    RewardLabel,
    TurnObservation,
)


def test_advice_block_defaults():
    advice = AdviceBlock(task_type="bugfix")
    assert advice.relevant_files == []
    assert advice.recommended_plan == []
    assert advice.confidence == 0.0


def test_advisor_input_packet_roundtrip():
    packet = AdvisorInputPacket(
        run_id="run-1",
        task_text="fix cli bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="task token match", score=0.8)],
        recent_failures=[FailureSignal(kind="wrong-file", summary="edited unrelated file")],
        constraints=["do not change public behavior"],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=1200,
        task=AdvisorTask(domain="coding", text="fix cli bug", type="bugfix"),
        context=AdvisorContext(
            summary="coding repo task",
            metadata={"repo": {"path": "/tmp/repo"}, "repo_summary": {"modules": ["app"]}},
        ),
        artifacts=[
            AdvisorArtifact(
                kind="file",
                locator="main.py",
                description="task token match",
                metadata={"score": 0.8},
                score=0.8,
            )
        ],
        history=[
            AdvisorHistoryEntry(
                kind="wrong-file",
                summary="edited unrelated file",
                metadata={"source": "recent_failures"},
            )
        ],
    )
    dumped = packet.model_dump()
    assert dumped["task_type"] == "bugfix"
    assert dumped["candidate_files"][0]["path"] == "main.py"
    assert dumped["task"]["domain"] == "coding"
    assert dumped["artifacts"][0]["locator"] == "main.py"
    assert dumped["history"][0]["kind"] == "wrong-file"


def test_advisor_input_packet_backfills_domain_capabilities():
    packet = AdvisorInputPacket(
        run_id="run-1",
        task_text="fix cli bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="task token match", score=0.8)],
        recent_failures=[],
        constraints=[],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=1200,
    )

    assert packet.domain_capabilities == [
        AdvisorCapabilityDescriptor(
            domain="coding",
            supported_artifact_kinds=["file"],
            supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
            supports_changed_artifacts=True,
            supports_symbol_regions=False,
        )
    ]


def test_advice_block_backfills_generic_focus_targets_from_compat_fields():
    advice = AdviceBlock(
        task_type="bugfix",
        relevant_files=[{"path": "main.py", "why": "entrypoint", "priority": 1}],
        relevant_symbols=[{"name": "main", "path": "main.py", "why": "dispatches execution"}],
    )

    assert advice.focus_targets == [
        FocusTarget(kind="file", locator="main.py", rationale="entrypoint", priority=1),
        FocusTarget(kind="symbol", locator="main.py::main", rationale="dispatches execution", priority=2),
    ]


def test_advice_block_backfills_compat_fields_from_generic_focus_targets():
    advice = AdviceBlock(
        task_type="research",
        focus_targets=[
            FocusTarget(kind="file", locator="docs/plan.md", rationale="source of truth", priority=1),
            FocusTarget(kind="symbol", locator="agent.py::run", rationale="execution entrypoint", priority=2),
        ],
    )

    assert [item.model_dump() for item in advice.relevant_files] == [
        {"path": "docs/plan.md", "why": "source of truth", "priority": 1}
    ]
    assert [item.model_dump() for item in advice.relevant_symbols] == [
        {"name": "run", "path": "agent.py", "why": "execution entrypoint"}
    ]


def test_advice_block_exposes_explicit_executor_injection_policy():
    advice = AdviceBlock(task_type="bugfix")

    assert advice.injection_policy == ExecutorInjectionPolicy(
        strategy="prepend",
        format="plain_text",
        min_confidence=0.0,
        include_confidence_note=True,
    )


def test_trajectory_models_round_trip_nested_turns():
    packet = AdvisorInputPacket(
        run_id="run-trajectory",
        task_text="fix the rollout loop",
        task_type="implementation",
        repo={"path": "/tmp/advisor", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["training"], hotspots=["training_rollouts.py"]),
        candidate_files=[CandidateFile(path="training_rollouts.py", reason="owns rollout loop", score=1.0)],
        token_budget=1200,
    )
    advice = AdviceBlock(task_type="implementation", recommended_plan=["advance one executor step"], confidence=0.9)
    observation = TurnObservation(
        turn_index=0,
        status="partial",
        executor_output="patched first step",
        summary="executor advanced one turn",
        files_touched=["training_rollouts.py"],
        tests_run=["pytest tests/agent/advisor/test_training_rollouts.py -q"],
        verifier_hints=["loop still needs stop policy"],
        error_messages=[],
        metrics={"tokens": 42},
    )
    trajectory = AdvisorTrajectory(
        trajectory_id="traj-1",
        run_id="run-trajectory",
        advisor_profile_id="generalist",
        task_text="fix the rollout loop",
        turns=[AdvisorTrajectoryTurn(turn_index=0, state_packet=packet, advice=advice, observation=observation)],
        final_outcome=AdvisorOutcome(run_id="run-trajectory", status="partial", summary="needs another turn"),
        final_reward=RewardLabel(run_id="run-trajectory", advisor_profile_id="generalist", total_reward=0.5, quality_score=0.5),
        stop_reason="max_turns",
        budget={"max_turns": 1},
    )

    restored = AdvisorTrajectory.model_validate(trajectory.model_dump())

    assert restored.turns[0].state_packet["task"]["domain"] == "coding"
    assert restored.turns[0].advice["recommended_plan"] == ["advance one executor step"]
    assert restored.turns[0].observation.files_touched == ["training_rollouts.py"]
    assert restored.final_reward["total_reward"] == 0.5
