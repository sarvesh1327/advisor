from agent.advisor.gateway import AdvisorGateway, create_app
from agent.advisor.schemas import AdviceBlock
from agent.advisor.settings import AdvisorSettings


class StubRuntime:
    def generate_advice(self, packet):
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect likely file"], confidence=0.7)


def test_gateway_builds_packet_and_returns_advice(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")

    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    gateway = AdvisorGateway(settings=settings, runtime=StubRuntime())
    result = gateway.task_run(task_text="fix main entrypoint bug", repo_path=str(repo), tool_limits={"write_allowed": True})

    assert result.advice_block.recommended_plan == ["inspect likely file"]
    assert result.advisor_input_packet.task_type == "bugfix"
    stored = gateway.trace_store.get_run(result.run_id)
    assert stored is not None


def test_create_app_uses_product_name(tmp_path):
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"))
    app = create_app(settings=settings)
    assert app.title == "Advisor"
