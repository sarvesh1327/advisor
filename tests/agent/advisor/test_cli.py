import json

from agent.advisor import cli
from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorTaskRunResult,
    CandidateFile,
    RepoSummary,
)


class StubGateway:
    def task_run(self, **kwargs):
        packet = AdvisorInputPacket(
            run_id="run-123",
            task_text=kwargs["task_text"],
            task_type="bugfix",
            repo={"path": kwargs["repo_path"], "branch": kwargs.get("branch"), "dirty": False},
            repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
            candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
            recent_failures=[],
            constraints=[],
            tool_limits=kwargs.get("tool_limits") or {},
            acceptance_criteria=kwargs.get("acceptance_criteria") or [],
            token_budget=1200,
        )
        return AdvisorTaskRunResult(
            run_id="run-123",
            advisor_input_packet=packet,
            advice_block=AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.9),
            model_version="advisor-qwen25-3b-v1",
            latency_ms=12,
        )


def test_cli_version_prints_version(capsys):
    exit_code = cli.main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "0.1.0"


def test_cli_run_prints_json(monkeypatch, tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.setattr(cli, "create_gateway", lambda **kwargs: StubGateway())
    exit_code = cli.main([
        "run",
        "--task-text",
        "fix main entrypoint bug",
        "--repo-path",
        str(repo),
        "--acceptance-criterion",
        "tests pass",
        "--tool-limit",
        "write_allowed=true",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["advisor_input_packet"]["acceptance_criteria"] == ["tests pass"]
    assert payload["advisor_input_packet"]["tool_limits"] == {"write_allowed": True}


def test_cli_serve_invokes_uvicorn(monkeypatch):
    calls = {}

    monkeypatch.setattr(cli, "create_http_app", lambda **kwargs: object())
    monkeypatch.setattr(cli, "uvicorn_run", lambda app, host, port: calls.update({"app": app, "host": host, "port": port}))

    exit_code = cli.main(["serve", "--host", "127.0.0.1", "--port", "9001"])

    assert exit_code == 0
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9001
