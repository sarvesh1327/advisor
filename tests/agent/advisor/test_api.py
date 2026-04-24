from fastapi.testclient import TestClient

from agent.advisor import create_gateway, create_http_app, get_version, run_task
from agent.advisor.core.schemas import AdviceBlock
from agent.advisor.core.settings import AdvisorSettings


class StubRuntime:
    def generate_advice(self, packet):
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect likely file"], confidence=0.7)


def test_run_task_uses_gateway_and_returns_result(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")

    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    gateway = create_gateway(settings=settings, runtime=StubRuntime())
    result = run_task(task_text="fix main entrypoint bug", repo_path=str(repo), gateway=gateway)

    assert result.advice_block.recommended_plan == ["inspect likely file"]
    assert result.advisor_input_packet.repo["path"] == str(repo)



def test_run_task_passes_profile_override_through_gateway(tmp_path):
    repo = tmp_path / "repo"
    (repo / "ui" / "layouts").mkdir(parents=True)
    (repo / "ui" / "layouts" / "home.json").write_text("{}")

    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    gateway = create_gateway(settings=settings, runtime=StubRuntime())
    result = run_task(
        task_text="refresh homepage layout from brief",
        repo_path=str(repo),
        gateway=gateway,
        advisor_profile_id="text-ui",
    )

    assert result.advisor_profile_id == "text-ui"
    assert result.advisor_input_packet.task.domain == "text-ui"


def test_create_http_app_includes_health_route(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_http_app(settings=settings)

    routes = {route.path for route in app.routes}
    assert "/" in routes
    assert "/healthz" in routes
    assert "/v1/advisor/task-run" in routes
    assert "/v1/operator/overview" in routes
    assert "/v1/operator/runs/{run_id}" in routes
    assert "/v1/operator/jobs" in routes
    assert "/v1/operator/jobs/{job_id}/resume" in routes
    assert "/v1/operator/jobs/{job_id}/run" in routes
    assert "/v1/operator/queue" in routes
    assert "/v1/operator/queue/pause" in routes
    assert "/v1/operator/queue/resume" in routes
    assert "/v1/operator/checkpoints/{advisor_profile_id}" in routes
    assert "/v1/operator/checkpoints/{advisor_profile_id}/{checkpoint_id}/eval" in routes
    assert "/v1/operator/retention/enforce" in routes
    assert "/v1/validation/gate" in routes
    assert "/v1/learning/controller" in routes
    assert "/v1/learning/controller/pause" in routes
    assert "/v1/learning/controller/resume" in routes
    assert "/v1/learning/readiness/{advisor_profile_id}" in routes
    assert "/v1/learning/profiles/{advisor_profile_id}/pause" in routes
    assert "/v1/learning/profiles/{advisor_profile_id}/resume" in routes
    assert "/v1/learning/profiles/{advisor_profile_id}/reset-backoff" in routes
    assert "/v1/learning/tick" in routes
    assert "/v1/operator/advisor-activity" in routes
    assert "/dashboard/advisor-activity" in routes


def test_force_eval_route_requires_and_preserves_benchmark_manifests(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    app = create_http_app(settings=settings)
    client = TestClient(app)
    benchmark_manifests = [
        {
            "run_id": "baseline-run",
            "fixture_id": "coding-main",
            "domain": "coding",
            "split": "validation",
            "packet_hash": "abc",
            "executor_config": {"name": "frontier-chat", "kind": "frontier_chat"},
            "verifier_set": ["build-check"],
            "routing_arm": "baseline",
            "reward_version": "phase8-v1",
            "score": {"overall_score": 0.5, "focus_target_recall": 0.5},
        },
        {
            "run_id": "advisor-run",
            "fixture_id": "coding-main",
            "domain": "coding",
            "split": "validation",
            "packet_hash": "abc",
            "executor_config": {"name": "frontier-chat", "kind": "frontier_chat"},
            "verifier_set": ["build-check"],
            "routing_arm": "advisor",
            "advisor_profile_id": "coding-default",
            "reward_version": "phase8-v1",
            "score": {"overall_score": 0.7, "focus_target_recall": 0.7},
        },
    ]

    response = client.post(
        "/v1/operator/checkpoints/coding-default/ckpt-1/eval",
        json={
            "benchmark_manifests": benchmark_manifests,
            "promotion_threshold": 0.2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["payload"]["candidate_checkpoint_id"] == "ckpt-1"
    assert payload["payload"]["promotion_threshold"] == 0.2
    assert payload["payload"]["benchmark_manifests"] == benchmark_manifests



def test_root_redirects_to_activity_dashboard(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_http_app(settings=settings)
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/dashboard/advisor-activity"


def test_get_version_returns_repo_version():
    assert get_version() == "0.1.0"


def test_validation_gate_route_reports_failed_jobs_and_missing_rollback(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    app = create_http_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/validation/gate",
        json={"required_profiles": ["coding-default"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pass"] is False
    assert "required_profiles" in payload["failed_checks"]
    assert payload["evidence"]["database_counts"]["runs"] == 0
    assert payload["evidence"]["database_counts"]["reward_labels"] == 0
    assert payload["evidence"]["artifact_counts"]["active_checkpoints"] == 0



def test_learning_controller_routes_expose_status_readiness_and_tick(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    app = create_http_app(settings=settings)
    client = TestClient(app)

    status_response = client.get("/v1/learning/controller")
    readiness_response = client.get("/v1/learning/readiness/coding-default")
    tick_response = client.post("/v1/learning/tick")

    assert status_response.status_code == 200
    assert status_response.json()["controller_paused"] is False
    assert readiness_response.status_code == 200
    assert readiness_response.json()["advisor_profile_id"] == "coding-default"
    assert tick_response.status_code == 200
    assert "readiness" in tick_response.json()
