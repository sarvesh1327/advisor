# Advisor production checklist

This is the working checklist for turning Advisor from a clean extracted prototype into a production-ready product.

## Phase 1 — Product surface
- [x] Add a stable Python API surface (`create_gateway`, `run_task`, `create_http_app`, `get_version`)
- [x] Add a real CLI entrypoint (`advisor`)
- [x] Add a minimal HTTP health endpoint
- [x] Expose version info consistently across API/CLI/HTTP
- [x] Document the public surface in the README

## Phase 2 — Config and runtime hardening
- [ ] Add structured config loading beyond env vars
- [ ] Validate config at startup
- [ ] Add runtime capability checks
- [ ] Add clearer missing-dependency and startup failures
- [ ] Add health/startup checks for model/runtime state

## Phase 3 — Packaging and repo hygiene
- [ ] Add CI for tests
- [ ] Add lint/format config
- [ ] Add CONTRIBUTING.md
- [ ] Add release workflow
- [ ] Pin/test supported Python versions

## Phase 4 — Inference robustness
- [ ] Add timeout handling
- [ ] Add malformed JSON recovery strategy
- [ ] Add retry policy for bad model output
- [ ] Add model warm-load path
- [ ] Add fallback behavior for smaller models

## Phase 5 — Context engine improvements
- [ ] Improve candidate file ranking
- [ ] Add changed-files awareness
- [ ] Add symbol extraction
- [ ] Improve recent-failure retrieval
- [ ] Handle larger repos more predictably

## Phase 6 — Evaluation
- [ ] Add golden eval fixtures
- [ ] Add replay harness for stored traces
- [ ] Measure no-advisor vs advisor behavior
- [ ] Track quality metrics beyond basic hit-rate
- [ ] Add regression-oriented eval docs

## Phase 7 — Data and training pipeline
- [ ] Add curation flow for exported traces
- [ ] Add quality scoring
- [ ] Add split policy (train/val/test)
- [ ] Add deduping / hard-case bucketing
- [ ] Add feedback loop notes for future tuning

## Phase 8 — Security, privacy, and observability
- [ ] Document stored data and retention expectations
- [ ] Add redaction/safe defaults guidance
- [ ] Add structured logs and run-id tracing
- [ ] Add metrics surface or export path
- [ ] Add production operating notes
