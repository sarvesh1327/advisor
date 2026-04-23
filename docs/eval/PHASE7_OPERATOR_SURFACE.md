# Phase 7 — Production operator surface

## Goal
Add first-class operator controls for queue management, checkpoint inspection, and forced profile evaluation over the existing product HTTP + CLI surfaces.

## Landed surfaces
- runtime helpers in `agent/advisor/operators/operator_runtime.py`
  - `inspect_profile_checkpoints(...)`
  - `enqueue_forced_profile_eval(...)`
  - persistent queue pause/resume state on `OperatorJobQueue`
- HTTP routes in `agent/advisor/product/gateway.py`
- CLI commands in `agent/advisor/product/cli.py`

## What Phase 7 now enables
- inspect operator queue state
- pause the operator queue with an explicit reason
- resume the operator queue safely
- prevent enqueue/run side effects while paused
- inspect checkpoints for a specific advisor profile
- enqueue a forced profile-local eval job for a chosen checkpoint

## Bounded guarantees from this phase
- queue controls are explicit and persistent
- checkpoint inspection is available without digging through raw registry files
- forced eval enqueueing is deterministic and operator-friendly
- both HTTP and CLI expose the same new operator control surfaces

## Verification
- targeted: `python -m pytest tests/agent/advisor/test_operator_runtime.py tests/agent/advisor/test_api.py tests/agent/advisor/test_cli.py -q`
- repo-wide: `ruff check .`
- full advisor suite: `python -m pytest tests/agent/advisor -q`

## Not implied yet
- Phase 8 final product validation / soak gate is not complete from this step alone
