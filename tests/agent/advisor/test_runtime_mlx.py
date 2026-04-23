import json
from types import SimpleNamespace

import pytest

from agent.advisor import runtime_mlx
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.runtime_mlx import MLXAdvisorRuntime
from agent.advisor.schemas import AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.settings import AdvisorSettings


class StubTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"


class CaptureTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "captured-prompt"


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



def test_runtime_builds_chat_prompt_without_assistant_prefill():
    tokenizer = CaptureTokenizer()
    runtime = MLXAdvisorRuntime(AdvisorSettings(system_prompt="You are a generic execution advisor."))

    prompt = runtime._build_generation_prompt(tokenizer, _packet())

    assert prompt == "captured-prompt"
    messages, kwargs = tokenizer.calls[0]
    assert messages[0]["content"] == "You are a generic execution advisor."
    assert [message["role"] for message in messages] == ["system", "user"]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}



def test_runtime_builds_chat_prompt_with_explicit_override():
    tokenizer = CaptureTokenizer()
    runtime = MLXAdvisorRuntime(AdvisorSettings(system_prompt="default generic prompt"))

    runtime._build_generation_prompt(tokenizer, _packet(), system_prompt="temporary override")

    messages, _kwargs = tokenizer.calls[0]
    assert messages[0]["content"] == "temporary override"



def test_runtime_prompt_includes_generic_packet_fields():
    runtime = MLXAdvisorRuntime(AdvisorSettings())

    prompt = runtime._format_prompt(_packet())

    assert "You are an execution advisor" in prompt
    assert "src/app/page.tsx" not in prompt
    assert "edit src/app/page.tsx" not in prompt
    assert "TASK_DOMAIN: coding" in prompt
    assert "ARTIFACTS:" in prompt
    assert "HISTORY:" in prompt
    assert "DOMAIN_CAPABILITIES:" in prompt
    assert "FOCUS_TARGETS" in prompt


def test_runtime_resolves_profile_training_spec_with_real_lora_metadata(tmp_path):
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        "\n".join(
            [
                'default_profile_id = "coding-default"',
                "",
                "[profiles.coding-default]",
                'domain = "coding"',
                'description = "Default coding advisor profile"',
                "",
                "[profiles.coding-default.training]",
                'backend = "grpo"',
                'base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"',
                'adapter_method = "lora"',
                "rollout_group_size = 4",
                "num_generations = 8",
                "max_steps = 12",
                "max_prompt_tokens = 4096",
                "max_completion_tokens = 1024",
                'checkpoint_root = "checkpoints/coding-default"',
                "lora_rank = 32",
                "lora_alpha = 64",
                "lora_dropout = 0.05",
                'target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]',
            ]
        ),
        encoding="utf-8",
    )
    registry = AdvisorProfileRegistry.from_toml(config_path)
    runtime = MLXAdvisorRuntime(AdvisorSettings())

    spec = runtime.resolve_profile_training_spec(registry, "coding-default")

    assert spec["base_model_name"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert spec["adapter_method"] == "lora"
    assert spec["lora_rank"] == 32
    assert spec["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_runtime_raises_clear_error_when_adapter_artifact_is_missing(tmp_path):
    runtime = MLXAdvisorRuntime(AdvisorSettings())
    checkpoint_dir = tmp_path / "checkpoints" / "coding-default-ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="adapter artifact"):
        runtime.resolve_adapter_artifact(checkpoint_dir)



def test_runtime_resolves_active_profile_adapter_metadata_from_promoted_checkpoint(tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "\n".join(
            [
                'default_profile_id = "coding-default"',
                "",
                "[profiles.coding-default]",
                'domain = "coding"',
                'description = "Default coding advisor profile"',
                "",
                "[profiles.coding-default.training]",
                'backend = "grpo"',
                'base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"',
                'adapter_method = "lora"',
                'rollout_group_size = 4',
                'num_generations = 8',
                'max_steps = 12',
                'max_prompt_tokens = 4096',
                'max_completion_tokens = 1024',
                'checkpoint_root = "checkpoints/coding-default"',
                'target_modules = ["q_proj"]',
                'lora_rank = 32',
            ]
        ),
        encoding="utf-8",
    )
    artifacts_root = tmp_path / "artifacts"
    checkpoint_dir = artifacts_root / "checkpoints" / "coding-default" / "coding-active"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapters.safetensors").write_bytes(b"adapter")
    (checkpoint_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name": "mlx-community/Qwen2.5-3B-Instruct-4bit"}, sort_keys=True),
        encoding="utf-8",
    )
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "coding-active",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {
                    "adapter_model": str(checkpoint_dir / "adapters.safetensors"),
                    "adapter_config": str(checkpoint_dir / "adapter_config.json"),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifacts_root / "checkpoint_registry.json").write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "coding-active",
                    "experiment_id": "exp-14",
                    "path": str(checkpoint_dir),
                    "status": "active",
                    "benchmark_summary": {},
                    "rollback_reason": None,
                    "advisor_profile_id": "coding-default",
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runtime = MLXAdvisorRuntime(
        AdvisorSettings(
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profiles_path=str(profiles_path),
            advisor_profile_id="coding-default",
        )
    )

    metadata = runtime.resolve_active_profile_adapter_metadata("coding-default")

    assert metadata["checkpoint_id"] == "coding-active"
    assert metadata["adapter_artifact_path"] == str(checkpoint_dir / "adapters.safetensors")
    assert metadata["base_model_name"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"



def test_runtime_loads_active_profile_adapter_when_present(monkeypatch, tmp_path):
    calls = []
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "\n".join(
            [
                'default_profile_id = "coding-default"',
                "",
                "[profiles.coding-default]",
                'domain = "coding"',
                'description = "Default coding advisor profile"',
                "",
                "[profiles.coding-default.training]",
                'backend = "grpo"',
                'base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"',
                'adapter_method = "lora"',
                'rollout_group_size = 4',
                'num_generations = 8',
                'max_steps = 12',
                'max_prompt_tokens = 4096',
                'max_completion_tokens = 1024',
                'checkpoint_root = "checkpoints/coding-default"',
                'target_modules = ["q_proj"]',
                'lora_rank = 32',
            ]
        ),
        encoding="utf-8",
    )
    artifacts_root = tmp_path / "artifacts"
    checkpoint_dir = artifacts_root / "checkpoints" / "coding-default" / "coding-active"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapters.safetensors").write_bytes(b"adapter")
    (checkpoint_dir / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora"}), encoding="utf-8")
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "coding-active",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {
                    "adapter_model": str(checkpoint_dir / "adapters.safetensors"),
                    "adapter_config": str(checkpoint_dir / "adapter_config.json"),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifacts_root / "checkpoint_registry.json").write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "coding-active",
                    "experiment_id": "exp-14",
                    "path": str(checkpoint_dir),
                    "status": "active",
                    "benchmark_summary": {},
                    "rollback_reason": None,
                    "advisor_profile_id": "coding-default",
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def fake_load(model_name, *, adapter_path=None):
        calls.append({"model_name": model_name, "adapter_path": adapter_path})
        return SimpleNamespace(), StubTokenizer()

    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", fake_load)
    runtime = MLXAdvisorRuntime(
        AdvisorSettings(
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profiles_path=str(profiles_path),
            advisor_profile_id="coding-default",
        )
    )

    runtime._ensure_loaded(advisor_profile_id="coding-default")

    assert calls == [
        {
            "model_name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "adapter_path": str(checkpoint_dir),
        }
    ]
    assert runtime.capabilities()["active_model_name"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"



def test_runtime_rejects_active_checkpoint_without_real_adapter_artifact(monkeypatch, tmp_path):
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        "\n".join(
            [
                'default_profile_id = "coding-default"',
                "",
                "[profiles.coding-default]",
                'domain = "coding"',
                'description = "Default coding advisor profile"',
                "",
                "[profiles.coding-default.training]",
                'backend = "grpo"',
                'base_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"',
                'adapter_method = "lora"',
                'rollout_group_size = 4',
                'num_generations = 8',
                'max_steps = 12',
                'max_prompt_tokens = 4096',
                'max_completion_tokens = 1024',
                'checkpoint_root = "checkpoints/coding-default"',
                'target_modules = ["q_proj"]',
                'lora_rank = 32',
            ]
        ),
        encoding="utf-8",
    )
    artifacts_root = tmp_path / "artifacts"
    checkpoint_dir = artifacts_root / "checkpoints" / "coding-default" / "coding-active"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "coding-active",
                "advisor_profile_id": "coding-default",
                "artifact_paths": {
                    "adapter_model": str(checkpoint_dir / "adapters.safetensors"),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifacts_root / "checkpoint_registry.json").write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "coding-active",
                    "experiment_id": "exp-14",
                    "path": str(checkpoint_dir),
                    "status": "active",
                    "benchmark_summary": {},
                    "rollback_reason": None,
                    "advisor_profile_id": "coding-default",
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", lambda *_args, **_kwargs: (SimpleNamespace(), StubTokenizer()))
    runtime = MLXAdvisorRuntime(
        AdvisorSettings(
            enable_fallback_runtime=False,
            trace_db_path=str(tmp_path / "advisor.db"),
            advisor_profiles_path=str(profiles_path),
            advisor_profile_id="coding-default",
        )
    )

    with pytest.raises(FileNotFoundError, match="adapter artifact"):
        runtime._ensure_loaded(advisor_profile_id="coding-default")



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



def test_runtime_timeout_does_not_wait_for_executor_shutdown(monkeypatch):
    calls = {}

    class FakeFuture:
        def result(self, timeout):
            calls["timeout"] = timeout
            raise runtime_mlx.FutureTimeoutError()

        def cancel(self):
            calls["cancelled"] = True
            return False

    class FakeExecutor:
        def __init__(self, max_workers):
            calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.shutdown()
            return False

        def submit(self, fn):
            calls["submitted"] = True
            return FakeFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            calls["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(runtime_mlx, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", lambda *_args, **_kwargs: "never returned")
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", lambda temp: {"temp": temp})

    runtime = MLXAdvisorRuntime(AdvisorSettings(inference_timeout_seconds=3))

    with pytest.raises(TimeoutError, match="generation exceeded timeout"):
        runtime._generate_response(SimpleNamespace(), StubTokenizer(), "prompt")

    assert calls["timeout"] == 3
    assert calls["cancelled"] is True
    assert calls["shutdown"] == (False, True)



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



def test_runtime_returns_heuristic_fallback_for_non_runtime_load_errors(monkeypatch):
    def fake_load(_model_name):
        raise OSError("model files missing")

    monkeypatch.setattr(runtime_mlx, "mlx_lm_load", fake_load)
    monkeypatch.setattr(runtime_mlx, "mlx_lm_generate", lambda *_args, **_kwargs: "unused")
    monkeypatch.setattr(runtime_mlx, "mlx_make_sampler", lambda temp: {"temp": temp})

    runtime = MLXAdvisorRuntime(AdvisorSettings(enable_fallback_runtime=True))
    advice = runtime.generate_advice(_packet())

    assert advice.task_type == "bugfix"
    assert advice.notes is not None
    assert "model files missing" in advice.notes
