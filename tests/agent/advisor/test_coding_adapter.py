from agent.advisor.coding_adapter import CodingContextAdapter
from agent.advisor.schemas import CandidateFile, FailureSignal


def test_coding_adapter_builds_packet_from_coding_inputs():
    adapter = CodingContextAdapter(token_budget=1200)

    packet = adapter.build_packet(
        run_id="run-1",
        task_text="fix main entrypoint bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        file_tree_slice=["main.py", "app/config.py"],
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[FailureSignal(kind="recent-failure", summary="edited unrelated file")],
        constraints=["avoid changing public API unless necessary"],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        changed_files=["main.py"],
    )

    assert packet.repo_summary.modules == ["main.py", "app"]
    assert packet.repo_summary.hotspots == ["main.py"]
    assert packet.task.domain == "coding"
    assert packet.artifacts[0].locator == "main.py"
    assert packet.artifacts[0].metadata["changed"] is True
    assert packet.history[0].kind == "recent-failure"
    assert packet.domain_capabilities[0].domain == "coding"


def test_coding_adapter_boosts_changed_code_and_excludes_build_outputs():
    adapter = CodingContextAdapter(token_budget=1200)

    packet = adapter.build_packet(
        run_id="run-2",
        task_text="fix the login handler bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        file_tree_slice=["src/login.py", "dist/login.js", "build/tmp.py"],
        candidate_files=[
            CandidateFile(path="dist/login.js", reason="filename overlap", score=1.0),
            CandidateFile(path="src/login.py", reason="changed implementation", score=0.9),
            CandidateFile(path="build/tmp.py", reason="compiled artifact", score=0.8),
        ],
        recent_failures=[],
        constraints=[],
        tool_limits={},
        acceptance_criteria=[],
        changed_files=["src/login.py"],
    )

    assert [candidate.path for candidate in packet.candidate_files] == ["src/login.py"]
    assert [artifact.locator for artifact in packet.artifacts] == ["src/login.py"]


def test_coding_adapter_exposes_symbol_hooks_for_source_files():
    adapter = CodingContextAdapter(token_budget=1200)

    packet = adapter.build_packet(
        run_id="run-3",
        task_text="fix login handler bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        file_tree_slice=["src/login_handler.py"],
        candidate_files=[CandidateFile(path="src/login_handler.py", reason="handler match", score=0.9)],
        recent_failures=[],
        constraints=[],
        tool_limits={},
        acceptance_criteria=[],
        changed_files=["src/login_handler.py"],
    )

    assert packet.domain_capabilities[0].supports_symbol_regions is True
    assert packet.artifacts[0].metadata["symbol_hint"] == "login_handler"
