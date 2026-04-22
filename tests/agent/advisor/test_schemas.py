import pytest

from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, FailureSignal, RepoSummary


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
    )
    dumped = packet.model_dump()
    assert dumped["task_type"] == "bugfix"
    assert dumped["candidate_files"][0]["path"] == "main.py"
