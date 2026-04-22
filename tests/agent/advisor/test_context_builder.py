from agent.advisor.context_builder import ContextBuilder
from agent.advisor.trace_store import AdvisorTraceStore


def test_context_builder_collects_repo_slice(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    (repo / "agent_prompt.py").write_text("PROMPT = 'x'\n")

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    builder = ContextBuilder(trace_store=store)
    packet = builder.build(
        task_text="fix prompt issue in main entrypoint",
        repo_path=str(repo),
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
    )

    assert packet.repo["path"] == str(repo)
    assert packet.repo_summary.file_tree_slice
    assert any(item.path.endswith("main.py") for item in packet.candidate_files)



def test_context_builder_ignores_next_build_artifacts(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".next" / "server" / "app").mkdir(parents=True)
    (repo / "src" / "app").mkdir(parents=True)
    (repo / ".next" / "server" / "app" / "page.js").write_text("compiled")
    (repo / "src" / "app" / "page.tsx").write_text("export default function Page() { return null }\n")
    (repo / "src" / "app" / "layout.tsx").write_text("export default function Layout({ children }) { return children }\n")

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    builder = ContextBuilder(trace_store=store)
    packet = builder.build(
        task_text="update the next app landing page UI",
        repo_path=str(repo),
        tool_limits={},
        acceptance_criteria=["build passes"],
    )

    assert all(not item.startswith(".next/") for item in packet.repo_summary.file_tree_slice)
    assert all(not candidate.path.startswith(".next/") for candidate in packet.candidate_files)
    assert packet.candidate_files[0].path == "src/app/page.tsx"
