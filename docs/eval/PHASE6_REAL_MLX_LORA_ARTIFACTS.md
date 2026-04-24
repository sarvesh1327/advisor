# Phase 6 — Real MLX LoRA artifact contract

## Status

Complete for the bounded Phase 6 slice in `docs/plans/2026-04-24-phase6-real-mlx-lora-artifacts.md`.

## Scope landed

- GRPO backend manifests now link the real MLX LoRA adapter layout:
  - `adapters.safetensors`
  - `adapter_config.json`
  - `checkpoint.json`
  - `backend-manifest.json`
- Backend manifests record Phase 6 provenance:
  - backend name
  - rollout group id
  - trajectory ids
  - advisor profile id / profile id
  - base model
  - target modules
  - LoRA rank
  - artifact paths
- Training runtime now records `training-manifest.json`, adds it to backend artifact paths, validates all required artifacts, and only then registers a candidate checkpoint.
- Missing required artifacts fail before candidate registration.

## Non-goals

- Runtime loading of promoted adapters remains out of scope for Phase 6.
- Promotion policy, dashboard/API/operator surfaces, multimodal training, dogfood, and autonomous training cycles were not changed.

## Verification

```bash
source .venv/bin/activate && python -m pytest tests/agent/advisor/test_real_training_backend.py::test_grpo_training_backend_manifest_records_real_lora_artifact_contract tests/agent/advisor/test_training_runtime.py::test_run_profile_training_job_refuses_candidate_when_adapter_file_is_missing tests/agent/advisor/test_training_runtime.py::test_run_profile_training_job_registers_candidate_only_after_required_artifacts_exist -q
# RED: 3 failed as expected before implementation.
# GREEN: 3 passed, 2 warnings.

source .venv/bin/activate && python -m pytest tests/agent/advisor/test_real_training_backend.py tests/agent/advisor/test_training_runtime.py -q
# 27 passed, 2 warnings.

source .venv/bin/activate && ruff check .
# All checks passed.

source .venv/bin/activate && python -m pytest tests/agent/advisor -q
# 222 passed, 2 warnings.
```

Advisor launchd service was intentionally kept stopped after verification because Sarvesh asked to stop Advisor middleware/service for now. `127.0.0.1:8765` was verified not listening before and after the full suite.
