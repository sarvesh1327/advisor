import json

from agent.advisor.core.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorTaskRunResult,
    CandidateFile,
    RepoSummary,
)
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.product import cli
from agent.advisor.storage.trace_store import AdvisorTraceStore

DEFAULT_PROFILE_ID = "coding-default"


class StubGateway:
    def __init__(self):
        self.calls = []

    def task_run(self, **kwargs):
        self.calls.append(kwargs)
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
            advisor_profile_id=kwargs.get("advisor_profile_id") or DEFAULT_PROFILE_ID,
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

    gateway = StubGateway()
    monkeypatch.setattr(cli, "create_gateway", lambda **kwargs: gateway)
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
        "--system-prompt",
        "You are a generic execution advisor.",
        "--advisor-profile-id",
        "image-ui",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "run-123"
    assert payload["advisor_profile_id"] == "image-ui"
    assert payload["advisor_input_packet"]["acceptance_criteria"] == ["tests pass"]
    assert payload["advisor_input_packet"]["tool_limits"] == {"write_allowed": True}
    assert gateway.calls[0]["system_prompt"] == "You are a generic execution advisor."
    assert gateway.calls[0]["advisor_profile_id"] == "image-ui"



def test_cli_serve_invokes_uvicorn(monkeypatch):
    calls = {}

    monkeypatch.setattr(cli, "create_http_app", lambda **kwargs: object())
    monkeypatch.setattr(cli, "uvicorn_run", lambda app, host, port: calls.update({"app": app, "host": host, "port": port}))

    exit_code = cli.main(["serve", "--host", "127.0.0.1", "--port", "9001"])

    assert exit_code == 0
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9001


def test_cli_operator_overview_prints_json(monkeypatch, tmp_path, capsys):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    store = AdvisorTraceStore(settings.trace_db_path)

    class StubGatewayWithStore:
        def __init__(self, trace_store):
            self.trace_store = trace_store

    monkeypatch.setattr(cli.AdvisorSettings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(cli, "create_gateway", lambda **kwargs: StubGatewayWithStore(store))

    exit_code = cli.main(["operator-overview"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["deployment"]["mode"] == "single_tenant"
    assert payload["live_metrics"]["total_runs"] == 0



def test_cli_operator_checkpoint_and_queue_controls_print_json(monkeypatch, tmp_path, capsys):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    store = AdvisorTraceStore(settings.trace_db_path)

    class StubGatewayWithStore:
        def __init__(self, trace_store):
            self.trace_store = trace_store
            self.profile_registry = None

    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "coding-default" / "ckpt-cli"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapters.safetensors").write_bytes(b"adapter")
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt-cli",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {"adapter_model": str(checkpoint_dir / "adapters.safetensors")},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry_path = tmp_path / "checkpoint_registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "ckpt-cli",
                    "experiment_id": "exp-cli",
                    "path": str(checkpoint_dir),
                    "status": "active",
                    "benchmark_summary": {"overall_score": 0.82},
                    "advisor_profile_id": "coding-default",
                    "rollback_reason": None,
                }
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.AdvisorSettings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(cli, "create_gateway", lambda **kwargs: StubGatewayWithStore(store))
    monkeypatch.setattr(cli, "_build_lifecycle_manager", lambda settings: cli.CheckpointLifecycleManager(tmp_path))

    pause_exit = cli.main(["operator-queue-pause", "--reason", "maintenance window"])
    pause_payload = json.loads(capsys.readouterr().out)
    queue_exit = cli.main(["operator-queue-status"])
    queue_payload = json.loads(capsys.readouterr().out)
    checkpoints_exit = cli.main(["operator-checkpoints", "--advisor-profile-id", "coding-default"])
    checkpoints_payload = json.loads(capsys.readouterr().out)
    resume_exit = cli.main(["operator-queue-resume"])
    resume_payload = json.loads(capsys.readouterr().out)
    force_eval_exit = cli.main(
        [
            "operator-force-eval",
            "--advisor-profile-id",
            "coding-default",
            "--checkpoint-id",
            "ckpt-cli",
            "--promotion-threshold",
            "0.2",
        ]
    )
    force_eval_payload = json.loads(capsys.readouterr().out)

    assert pause_exit == 0
    assert pause_payload["paused"] is True
    assert queue_exit == 0
    assert queue_payload["paused"] is True
    assert checkpoints_exit == 0
    assert checkpoints_payload[0]["checkpoint_id"] == "ckpt-cli"
    assert resume_exit == 0
    assert resume_payload["paused"] is False
    assert force_eval_exit == 0
    assert force_eval_payload["job_type"] == "eval-profile"
    assert force_eval_payload["payload"]["candidate_checkpoint_id"] == "ckpt-cli"
    assert force_eval_payload["payload"]["promotion_threshold"] == 0.2



def test_cli_deployment_profile_respects_mode_override(monkeypatch, tmp_path, capsys):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(cli.AdvisorSettings, "load", classmethod(lambda cls: settings))

    exit_code = cli.main(["deployment-profile", "--mode", "hosted"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "hosted"
    assert payload["auth_boundary"] == "external auth proxy required"


def test_cli_hardening_profile_and_release_gate_commands(monkeypatch, tmp_path, capsys):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(cli.AdvisorSettings, "load", classmethod(lambda cls: settings))
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "canonical_study": {"lift_summary": {"advisor_minus_baseline_overall_score": 0.1}},
                "provenance_coverage": {"reward_label_coverage": 1.0, "lineage_coverage": 1.0},
                "paper_divergences": [{"area": "executor_surface", "status": "intentional_productization"}],
            }
        ),
        encoding="utf-8",
    )

    hardening_exit = cli.main(["hardening-profile", "--mode", "hosted"])
    hardening_payload = json.loads(capsys.readouterr().out)
    release_exit = cli.main(["release-gate", "--report-path", str(report_path)])
    release_payload = json.loads(capsys.readouterr().out)

    assert hardening_exit == 0
    assert hardening_payload["mode"] == "hosted"
    assert hardening_payload["tenancy"]["tenant_id_required"] is True
    assert release_exit == 0
    assert release_payload["verdict"]["pass"] is True
    assert release_payload["alerts"]["severity"] == "info"


def test_cli_export_and_import_bundle_commands(monkeypatch, tmp_path, capsys):
    state_root = tmp_path / "state"
    state_root.mkdir()
    trace_db = state_root / "advisor.db"
    event_log = state_root / "events.jsonl"
    trace_db.write_text("sqlite-placeholder", encoding="utf-8")
    event_log.write_text('{"event":"ok"}\n', encoding="utf-8")
    settings = AdvisorSettings(enabled=True, trace_db_path=str(trace_db), event_log_path=str(event_log))
    monkeypatch.setattr(cli.AdvisorSettings, "load", classmethod(lambda cls: settings))

    export_exit = cli.main(["export-bundle", "--output-dir", str(tmp_path / "bundle")])
    export_payload = json.loads(capsys.readouterr().out)
    import_exit = cli.main(
        [
            "import-bundle",
            "--bundle-path",
            str(tmp_path / "bundle"),
            "--target-root",
            str(tmp_path / "restored"),
        ]
    )
    import_payload = json.loads(capsys.readouterr().out)

    assert export_exit == 0
    assert (tmp_path / "bundle" / "state" / "advisor.db").exists()
    assert import_exit == 0
    assert (tmp_path / "restored" / "state" / "advisor.db").exists()
    assert export_payload["bundle_path"] == str(tmp_path / "bundle")
    assert import_payload["restored_root"] == str(tmp_path / "restored")
