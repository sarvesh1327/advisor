from agent.advisor.gateway import AdvisorGateway, create_app
from agent.advisor.schemas import AdviceBlock
from agent.advisor.settings import AdvisorSettings


class StubRuntime:
    def __init__(self):
        self.warmup_calls = 0
        self.generate_calls = []

    def generate_advice(self, packet, system_prompt=None):
        self.generate_calls.append({"packet": packet, "system_prompt": system_prompt})
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect likely file"], confidence=0.7)

    def capabilities(self):
        return {"runtime": "stub", "available": True, "ready": True, "reason": None}

    def warmup(self):
        self.warmup_calls += 1


class UnavailableRuntime:
    def generate_advice(self, packet):
        raise AssertionError("should not be called in health tests")

    def capabilities(self):
        return {"runtime": "stub", "available": False, "ready": False, "reason": "model files missing"}



def test_gateway_builds_packet_and_returns_advice(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")

    runtime = StubRuntime()
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    gateway = AdvisorGateway(settings=settings, runtime=runtime)
    result = gateway.task_run(
        task_text="fix main entrypoint bug",
        repo_path=str(repo),
        tool_limits={"write_allowed": True},
        system_prompt="You are a generic execution advisor.",
    )

    assert result.advice_block.recommended_plan == ["inspect likely file"]
    assert result.advisor_input_packet.task_type == "bugfix"
    assert runtime.generate_calls[0]["system_prompt"] == "You are a generic execution advisor."
    stored = gateway.trace_store.get_run(result.run_id)
    assert stored is not None



def test_gateway_warm_load_calls_runtime_warmup(tmp_path):
    runtime = StubRuntime()
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), warm_load_on_start=True)

    AdvisorGateway(settings=settings, runtime=runtime)

    assert runtime.warmup_calls == 1



def test_create_app_uses_product_name(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_app(settings=settings)
    assert app.title == "Advisor"



def test_gateway_system_health_reports_runtime_status(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    gateway = AdvisorGateway(settings=settings, runtime=UnavailableRuntime())

    health = gateway.system_health()

    assert health["status"] == "degraded"
    assert health["runtime"]["available"] is False
    assert health["config"]["valid"] is True



def test_health_route_returns_gateway_health(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_app(settings=settings, runtime=UnavailableRuntime())
    health_endpoint = next(route.endpoint for route in app.routes if route.path == "/healthz")

    payload = health_endpoint()

    assert payload["status"] == "degraded"
    assert payload["runtime"]["reason"] == "model files missing"
