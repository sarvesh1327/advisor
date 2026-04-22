from types import SimpleNamespace

import pytest

from agent.advisor import runtime_mlx
from agent.advisor.runtime_mlx import MLXAdvisorRuntime
from agent.advisor.schemas import AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.settings import AdvisorSettings


class StubTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"



def _packet() -> AdvisorInputPacket:
    return AdvisorInputPacket(
        run_id="run-1",
        task_text="fix main entrypoint bug",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=[],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=1200,
    )



def test_runtime_raises_clear_error_when_mlx_lm_unavailable(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", None)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", None)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", None)

    runtime = MLXAdvisorRuntime(AdvisorSettings())

    with pytest.raises(RuntimeError, match="mlx-lm"):
        runtime._ensure_loaded()



def test_runtime_capabilities_report_missing_dependencies(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", None)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", None)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", None)

    runtime = MLXAdvisorRuntime(AdvisorSettings(enable_fallback_runtime=False))
    capabilities = runtime.capabilities()

    assert capabilities["runtime"] == "mlx"
    assert capabilities["available"] is False
    assert capabilities["ready"] is False
    assert "mlx-lm" in capabilities["reason"]



def test_runtime_retries_after_malformed_json(monkeypatch):
    calls = {"count": 0}

    def fake_load(_model_name):
        return SimpleNamespace(), StubTokenizer()

    def fake_generate(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return "not json"
        return '{"task_type":"bugfix","recommended_plan":["inspect main.py"],"confidence":0.8}'

    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", fake_load)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", fake_generate)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", lambda temp: {"temp": temp})

    runtime = MLXAdvisorRuntime(AdvisorSettings(max_retries=1))
    advice = runtime.generate_advice(_packet())

    assert advice.recommended_plan == ["inspect main.py"]
    assert calls["count"] == 2



def test_runtime_raises_timeout_error_when_generation_exceeds_limit(monkeypatch):
    def fake_load(_model_name):
        return SimpleNamespace(), StubTokenizer()

    def fake_generate(*_args, **_kwargs):
        raise TimeoutError("generation exceeded timeout")

    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", fake_load)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", fake_generate)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", lambda temp: {"temp": temp})

    runtime = MLXAdvisorRuntime(AdvisorSettings(max_retries=0, enable_fallback_runtime=False))

    with pytest.raises(TimeoutError):
        runtime.generate_advice(_packet())



def test_runtime_warmup_marks_runtime_ready(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", lambda _model_name: (SimpleNamespace(), StubTokenizer()))

    runtime = MLXAdvisorRuntime(AdvisorSettings())
    runtime.warmup()
    capabilities = runtime.capabilities()

    assert capabilities["available"] is True
    assert capabilities["ready"] is True



def test_runtime_uses_fallback_model_when_primary_load_fails(monkeypatch):
    loaded_models = []

    def fake_load(model_name):
        loaded_models.append(model_name)
        if model_name == "primary-model":
            raise RuntimeError("primary missing")
        return SimpleNamespace(), StubTokenizer()

    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", fake_load)

    runtime = MLXAdvisorRuntime(
        AdvisorSettings(model_name="primary-model", fallback_model_name="fallback-model")
    )
    runtime.warmup()

    assert loaded_models == ["primary-model", "fallback-model"]
    assert runtime.capabilities()["active_model_name"] == "fallback-model"



def test_runtime_returns_heuristic_fallback_advice_when_enabled(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", None)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", None)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", None)

    runtime = MLXAdvisorRuntime(AdvisorSettings(enable_fallback_runtime=True))
    advice = runtime.generate_advice(_packet())

    assert advice.task_type == "bugfix"
    assert "main.py" in advice.recommended_plan[0]
    assert advice.notes is not None
