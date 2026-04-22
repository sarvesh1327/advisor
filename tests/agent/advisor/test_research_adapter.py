from agent.advisor.research_adapter import ResearchContextAdapter
from agent.advisor.schemas import CandidateFile, FailureSignal


def test_research_adapter_builds_packet_from_sources_and_notes():
    adapter = ResearchContextAdapter(token_budget=900)

    packet = adapter.build_packet(
        run_id="run-r1",
        task_text="draft a launch brief grounded in the source notes",
        task_type="research",
        repo={"path": "/tmp/research", "branch": "main", "dirty": False},
        file_tree_slice=["docs/brief.md", "notes/interview.txt", "sources/metrics.md"],
        candidate_files=[
            CandidateFile(path="docs/brief.md", reason="target brief", score=0.9),
            CandidateFile(path="notes/interview.txt", reason="supporting notes", score=0.6),
        ],
        recent_failures=[
            FailureSignal(
                kind="recent-failure",
                file="notes/interview.txt",
                summary="previous draft made unsupported claims",
                fix_hint="cite the interview notes",
            )
        ],
        constraints=["preserve citations"],
        tool_limits={"web_read": True},
        acceptance_criteria=["brief cites sources"],
        changed_files=["docs/brief.md"],
    )

    assert packet.task.domain == "research-writing"
    assert packet.context.summary == "research-writing task context"
    assert packet.artifacts[0].locator == "docs/brief.md"
    assert packet.artifacts[0].metadata["changed"] is True
    assert packet.artifacts[1].kind == "note"
    assert packet.history[0].summary == "previous draft made unsupported claims"
    assert packet.domain_capabilities[0].domain == "research-writing"
    assert packet.domain_capabilities[0].supported_artifact_kinds == ["document", "note", "source"]


def test_research_adapter_prefers_notes_and_sources_over_stale_drafts():
    adapter = ResearchContextAdapter(token_budget=900)

    packet = adapter.build_packet(
        run_id="run-r2",
        task_text="summarize the source notes for the launch memo",
        task_type="research",
        repo={"path": "/tmp/research", "branch": "main", "dirty": False},
        file_tree_slice=["docs/brief.md", "notes/interview.txt", "sources/metrics.md"],
        candidate_files=[
            CandidateFile(path="docs/brief.md", reason="existing memo draft", score=0.8),
            CandidateFile(path="notes/interview.txt", reason="interview notes", score=0.6),
            CandidateFile(path="sources/metrics.md", reason="metrics source", score=0.55),
        ],
        recent_failures=[],
        constraints=[],
        tool_limits={},
        acceptance_criteria=[],
        changed_files=[],
    )

    assert [candidate.path for candidate in packet.candidate_files] == [
        "notes/interview.txt",
        "sources/metrics.md",
        "docs/brief.md",
    ]
