from agent.advisor.gateway import AdvisorGateway, create_app
from agent.advisor.schemas import AdviceBlock, ExecutorInjectionPolicy
from agent.advisor.settings import AdvisorSettings


def _write_profiles_config(path):
    path.write_text(
        """
        default_profile_id = "coding-default"

        [profiles.coding-default]
        domain = "coding"
        description = "Default coding advisor profile"

        [profiles.image-ui]
        domain = "image-ui"
        description = "UI advisor profile"
        """.strip(),
        encoding="utf-8",
    )


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
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)

    runtime = StubRuntime()
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        advisor_profile_id="coding-default",
        advisor_profiles_path=str(profiles_path),
    )
    gateway = AdvisorGateway(settings=settings, runtime=runtime)
    result = gateway.task_run(
        task_text="fix main entrypoint bug",
        repo_path=str(repo),
        tool_limits={"write_allowed": True},
        system_prompt="You are a generic execution advisor.",
    )

    assert result.advice_block.recommended_plan == ["inspect likely file"]
    assert result.advisor_input_packet.task_type == "bugfix"
    assert result.advisor_profile_id == "coding-default"
    assert runtime.generate_calls[0]["system_prompt"] == "You are a generic execution advisor."
    stored = gateway.trace_store.get_run(result.run_id)
    assert stored is not None
    assert stored["advisor_profile_id"] == "coding-default"
    assert stored["injected_advice"]["recommended_plan"] == ["inspect likely file"]
    assert stored["injected_rendered_advice"].startswith("[Advisor hint")
    assert stored["injection_policy"]["strategy"] == "prepend"


def test_gateway_resolves_explicit_profile_override(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)

    gateway = AdvisorGateway(
        settings=AdvisorSettings(
            enabled=True,
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profile_id="coding-default",
            advisor_profiles_path=str(profiles_path),
        ),
        runtime=StubRuntime(),
    )

    result = gateway.task_run(
        task_text="fix main entrypoint bug",
        repo_path=str(repo),
        advisor_profile_id="image-ui",
    )
    stored = gateway.trace_store.get_run(result.run_id)

    assert result.advisor_profile_id == "image-ui"
    assert stored is not None
    assert stored["advisor_profile_id"] == "image-ui"


class ProfileAwareStubRuntime(StubRuntime):
    def generate_advice(self, packet, system_prompt=None, advisor_profile_id=None):
        self.generate_calls.append(
            {
                "packet": packet,
                "system_prompt": system_prompt,
                "advisor_profile_id": advisor_profile_id,
            }
        )
        return AdviceBlock(task_type=packet.task_type, recommended_plan=["inspect likely file"], confidence=0.7)



def test_gateway_passes_resolved_profile_to_profile_aware_runtime(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)
    runtime = ProfileAwareStubRuntime()
    gateway = AdvisorGateway(
        settings=AdvisorSettings(
            enabled=True,
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profile_id="coding-default",
            advisor_profiles_path=str(profiles_path),
        ),
        runtime=runtime,
    )

    gateway.task_run(
        task_text="fix main entrypoint bug",
        repo_path=str(repo),
        advisor_profile_id="image-ui",
    )

    assert runtime.generate_calls[0]["advisor_profile_id"] == "image-ui"



def test_gateway_respects_explicit_injection_policy_threshold(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def main():\n    pass\n")
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)

    class LowConfidenceRuntime(StubRuntime):
        def generate_advice(self, packet, system_prompt=None):
            self.generate_calls.append({"packet": packet, "system_prompt": system_prompt})
            return AdviceBlock(
                task_type=packet.task_type,
                recommended_plan=["inspect likely file"],
                confidence=0.2,
                injection_policy=ExecutorInjectionPolicy(min_confidence=0.5),
            )

    gateway = AdvisorGateway(
        settings=AdvisorSettings(
            enabled=True,
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profile_id="coding-default",
            advisor_profiles_path=str(profiles_path),
        ),
        runtime=LowConfidenceRuntime(),
    )

    result = gateway.task_run(task_text="fix main entrypoint bug", repo_path=str(repo))
    stored = gateway.trace_store.get_run(result.run_id)

    assert stored is not None
    assert "below injection threshold" in stored["injected_rendered_advice"]



def test_gateway_warm_load_calls_runtime_warmup(tmp_path):
    runtime = StubRuntime()
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        warm_load_on_start=True,
        advisor_profile_id="coding-default",
        advisor_profiles_path=str(profiles_path),
    )

    AdvisorGateway(settings=settings, runtime=runtime)

    assert runtime.warmup_calls == 1



def test_create_app_uses_product_name(tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        advisor_profile_id="coding-default",
        advisor_profiles_path=str(profiles_path),
    )
    app = create_app(settings=settings)
    assert app.title == "Advisor"



def test_gateway_system_health_reports_runtime_status(tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        advisor_profile_id="coding-default",
        advisor_profiles_path=str(profiles_path),
    )
    gateway = AdvisorGateway(settings=settings, runtime=UnavailableRuntime())

    health = gateway.system_health()

    assert health["status"] == "degraded"
    assert health["runtime"]["available"] is False
    assert health["config"]["valid"] is True
    assert health["config"]["advisor_profile_id"] == "coding-default"



def test_health_route_returns_gateway_health(tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    _write_profiles_config(profiles_path)
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        advisor_profile_id="coding-default",
        advisor_profiles_path=str(profiles_path),
    )
    app = create_app(settings=settings, runtime=UnavailableRuntime())
    health_endpoint = next(route.endpoint for route in app.routes if route.path == "/healthz")

    payload = health_endpoint()

    assert payload["status"] == "degraded"
    assert payload["runtime"]["reason"] == "model files missing"
