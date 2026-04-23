from agent.advisor.adapters.context_builder import ContextBuilder
from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome
from agent.advisor.storage.trace_store import AdvisorTraceStore


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


def test_context_builder_delegates_packet_construction_to_adapter(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")

    calls = {}

    class StubAdapter:
        def build_packet(self, **kwargs):
            calls.update(kwargs)
            return {"built": True, "run_id": kwargs["run_id"]}

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    builder = ContextBuilder(trace_store=store, packet_adapter=StubAdapter())

    packet = builder.build(
        task_text="fix main entrypoint bug",
        repo_path=str(repo),
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        run_id="run-123",
    )

    assert packet == {"built": True, "run_id": "run-123"}
    assert calls["task_text"] == "fix main entrypoint bug"
    assert calls["repo"]["path"] == str(repo)
    assert calls["candidate_files"][0].path == "main.py"


def test_context_builder_passes_changed_files_to_adapter(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs").mkdir()
    (repo / "docs" / "brief.md").write_text("draft\n")

    calls = {}

    class StubAdapter:
        def build_packet(self, **kwargs):
            calls.update(kwargs)
            return {"changed_files": kwargs["changed_files"]}

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    builder = ContextBuilder(trace_store=store, packet_adapter=StubAdapter())

    packet = builder.build(
        task_text="update the launch brief",
        repo_path=str(repo),
        tool_limits={},
        acceptance_criteria=["brief is updated"],
        changed_files=["docs/brief.md"],
    )

    assert packet == {"changed_files": ["docs/brief.md"]}
    assert calls["changed_files"] == ["docs/brief.md"]


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
    assert packet.task.domain == "coding"
    assert packet.context.metadata["repo"]["path"] == str(repo)
    assert packet.artifacts[0].kind == "file"
    assert packet.artifacts[0].locator == "src/app/page.tsx"


def test_context_builder_selects_image_adapter_for_ui_tasks(tmp_path):
    repo = tmp_path / "repo"
    (repo / "ui" / "mockups").mkdir(parents=True)
    (repo / "ui" / "layouts").mkdir(parents=True)
    (repo / "ui" / "mockups" / "home.png").write_text("image")
    (repo / "ui" / "layouts" / "home.json").write_text("{}")

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    builder = ContextBuilder(trace_store=store)
    packet = builder.build(
        task_text="refresh the hero image to match the updated mockup",
        repo_path=str(repo),
        tool_limits={"image_read": True},
        acceptance_criteria=["hero matches mock"],
        changed_files=["ui/mockups/home.png"],
    )

    assert packet.task.domain == "image-ui"
    assert packet.domain_capabilities[0].domain == "image-ui"
    assert packet.artifacts[0].kind == "image"


def test_context_builder_packs_large_context_to_budget(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    for index in range(6):
        (repo / "src" / f"feature_{index}.py").write_text(f"def feature_{index}():\n    return {index}\n")

    store = AdvisorTraceStore(tmp_path / "advisor.db")
    for index in range(4):
        store.record_task_run(
            AdvisorInputPacket(
                run_id=f"run-old-{index}",
                task_text=f"fix feature_{index} regression",
                task_type="bugfix",
                repo={"path": str(repo), "branch": "main", "dirty": False},
                repo_summary={"modules": ["src"], "hotspots": [f"src/feature_{index}.py"], "file_tree_slice": [f"src/feature_{index}.py"]},
                candidate_files=[{"path": f"src/feature_{index}.py", "reason": "match", "score": 0.7}],
                recent_failures=[],
                constraints=[],
                tool_limits={},
                acceptance_criteria=[],
                token_budget=900,
            ),
            AdviceBlock(task_type="bugfix", recommended_plan=["inspect file"], confidence=0.5),
            advisor_model="advisor-test",
            latency_ms=5,
            prompt_hash=f"hash-{index}",
        )
        store.record_outcome(
            AdvisorOutcome(
                run_id=f"run-old-{index}",
                status="failure",
                files_touched=[f"src/feature_{index}.py"],
                tests_run=["pytest -q"],
                summary=f"feature_{index} still failing",
            )
        )

    builder = ContextBuilder(trace_store=store, max_candidate_files=6, max_failures=4, token_budget=320)
    packet = builder.build(
        task_text="fix feature regression",
        repo_path=str(repo),
        tool_limits={},
        acceptance_criteria=["tests pass"],
        changed_files=["src/feature_0.py"],
    )

    assert len(packet.candidate_files) == 1
    assert len(packet.artifacts) == 1
    assert len(packet.history) == 0
    assert packet.context.metadata["packed"] == {"candidate_files": 1, "artifacts": 1, "history": 0}
