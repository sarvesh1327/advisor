# Phase 8 — Final validation gate

## Goal
Add one honest final validation / soak gate over the existing checkpoint lineage and operator job history so the product can report whether it has enough evidence to be considered validated.

## Landed surfaces
- reusable report/gate in `agent/advisor/product/hardening.py`
  - `Phase8ValidationPolicy`
  - `build_phase8_validation_report(...)`
- HTTP route in `agent/advisor/product/gateway.py`
  - `POST /v1/validation/gate`
- CLI command in `agent/advisor/product/cli.py`
  - `validation-gate`
- public exports in `agent/advisor/__init__.py`

## What the gate checks
Per required profile:
- completed cycle count
- promoted cycle count
- active checkpoint presence
- best observed positive improvement
- rollback evidence coverage

Global:
- failed operator job count against policy
- missing required profile coverage

## Bounded guarantees from this phase
- final product validation is no longer a manual judgment over scattered reports
- the gate is explicit about what evidence is missing
- callers can choose which profiles must be validated before claiming readiness
- the same report is available through library, HTTP, and CLI surfaces

## Verification
- targeted: `python -m pytest tests/agent/advisor/test_validation_gate.py tests/agent/advisor/test_api.py tests/agent/advisor/test_cli.py -q`
- repo-wide: `ruff check .`
- full advisor suite: `python -m pytest tests/agent/advisor -q`

## Not implied yet
- this does not create new soak data by itself
- this does not replace the legacy Phase 16 benchmark release gate
- this gate is only as strong as the operator history and checkpoint lineage present in state
