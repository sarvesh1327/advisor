from agent.advisor import create_gateway, create_http_app, get_version, run_task
from agent.advisor.schemas import AdviceBlock
from agent.advisor.settings import AdvisorSettings


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


def test_create_http_app_includes_health_route(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_http_app(settings=settings)

    routes = {route.path for route in app.routes}
    assert "/healthz" in routes
    assert "/v1/advisor/task-run" in routes
    assert "/v1/operator/overview" in routes
    assert "/v1/operator/runs/{run_id}" in routes
    assert "/v1/operator/jobs" in routes
    assert "/v1/operator/jobs/{job_id}/resume" in routes
    assert "/v1/operator/retention/enforce" in routes


def test_get_version_returns_repo_version():
    assert get_version() == "0.1.0"
