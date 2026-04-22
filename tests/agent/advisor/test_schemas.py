
from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorArtifact,
    AdvisorCapabilityDescriptor,
    AdvisorContext,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorTask,
    CandidateFile,
    FailureSignal,
    RepoSummary,
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
