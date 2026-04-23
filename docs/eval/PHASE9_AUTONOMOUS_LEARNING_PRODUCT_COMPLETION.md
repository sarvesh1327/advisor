# Phase 9 — Autonomous learning product completion

## Goal
Add the missing always-on autonomous learning product layer so Advisor is code-complete as a self-running learning system, not just a callable manual loop.

## Landed surfaces
- new `agent/advisor/learning/` package
  - `state.py`
  - `readiness.py`
  - `controller.py`
  - `service.py`
- extended validation gate in `agent/advisor/product/hardening.py`
- HTTP learning routes in `agent/advisor/product/gateway.py`
- CLI learning commands in `agent/advisor/product/cli.py`
- public exports in `agent/advisor/__init__.py`

## What Phase 9 now enables
- persistent autonomous controller state
- per-profile readiness reports from real stored runs
- fresh-run ingestion into rollout groups
- autonomous train/eval/promote tick orchestration
- per-profile pause/resume/reset-backoff controls
- controller pause/resume controls
- bounded autonomous learning service loop
- validation gate visibility for autonomous-learning readiness

## Mandatory dogfood evidence
Advisor was run against the Advisor repo itself through the orchestrator/live-run path, not just the lightweight gateway advice path.

Dogfood state root:
- `tmp/phase9-dogfood-state`

Observed dogfood run ids:
- `run-dogfood-a`
- `run-dogfood-b`

Observed reward totals:
- `run-dogfood-a` → `0.975`
- `run-dogfood-b` → `0.925`

Observed autonomous ingestion outcome:
- coding-default readiness became `ready=true`
- controller consumed both dogfood runs
- controller launched one autonomous cycle for `coding-default`
- controller persisted completed-cycle metadata and active job ids

## Bounded guarantees from this phase
- the product now has a first-class autonomous learning controller/service
- dogfood runs with lineage + reward labels can feed autonomous training ingestion
- autonomous state survives restart via persisted controller state
- operators can inspect and control the controller/profile lifecycle from API + CLI
- remaining caveat after this phase is runtime evidence duration, not missing product code

## Verification
- targeted:
  - `python -m pytest tests/agent/advisor/test_learning_controller.py tests/agent/advisor/test_learning_service.py tests/agent/advisor/test_validation_gate.py tests/agent/advisor/test_api.py tests/agent/advisor/test_cli.py -q`
- repo-wide:
  - `ruff check .`
- full advisor suite:
  - `python -m pytest tests/agent/advisor -q`
