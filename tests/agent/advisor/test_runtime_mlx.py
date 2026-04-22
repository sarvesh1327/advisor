from agent.advisor import runtime_mlx
from agent.advisor.runtime_mlx import MLXAdvisorRuntime
from agent.advisor.settings import AdvisorSettings



def test_runtime_raises_clear_error_when_mlx_lm_unavailable(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", None)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", None)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", None)

    runtime = MLXAdvisorRuntime(AdvisorSettings())

    try:
        runtime._ensure_loaded()
    except RuntimeError as exc:
        assert "mlx-lm" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when mlx-lm is unavailable")



def test_runtime_capabilities_report_missing_dependencies(monkeypatch):
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", None)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", None)
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", None)

    runtime = MLXAdvisorRuntime(AdvisorSettings())
    capabilities = runtime.capabilities()

    assert capabilities["runtime"] == "mlx"
    assert capabilities["available"] is False
    assert capabilities["ready"] is False
    assert "mlx-lm" in capabilities["reason"]
