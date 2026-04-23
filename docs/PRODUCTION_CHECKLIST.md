# Advisor production checklist

This is the working checklist for turning Advisor into a generic, paper-faithful implementation of How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models, including the reward system needed to improve the advisor over time.

Core rule:
- every major implementation decision should strengthen the paper loop: packet -> advisor advice -> black-box executor -> outcome/reward -> replay/eval -> advisor improvement
- avoid coding-only assumptions in core abstractions unless they are explicitly isolated behind a domain adapter

## Phase 1 — Product surface
- [x] Add a stable Python API surface (`create_gateway`, `run_task`, `create_http_app`, `get_version`)
- [x] Add a real CLI entrypoint (`advisor`)
- [x] Add a minimal HTTP health endpoint
- [x] Expose version info consistently across API/CLI/HTTP
- [x] Document the public surface in the README

## Phase 2 — Config and runtime hardening
- [x] Add structured config loading beyond env vars
- [x] Validate config at startup
- [x] Add runtime capability checks
- [x] Add clearer missing-dependency and startup failures
- [x] Add health/startup checks for model/runtime state

## Phase 3 — Packaging and repo hygiene
- [x] Add CI for tests
- [x] Add lint/format config
- [x] Add CONTRIBUTING.md
- [x] Add release workflow
- [x] Pin/test supported Python versions

## Phase 4 — Inference robustness
- [x] Add timeout handling
- [x] Add malformed JSON recovery strategy
- [x] Add retry policy for bad model output
- [x] Add model warm-load path
- [x] Add fallback behavior for smaller models

## Phase 5 — Generic packet and context engine
- [ ] Split core advisor abstractions from coding-specific packet builders
- [ ] Make the generic packet (`task`, `context`, `artifacts`, `constraints`, `history`, `acceptance_criteria`, `domain_capabilities`) the canonical packet surface
- [ ] Keep coding-only packet fields as compatibility fields or adapter extensions, not the primary abstraction
- [x] Add a coding adapter that maps repos/files/failures into the generic packet
- [x] Add an image-or-UI adapter that maps references/screenshots/layout constraints into the generic packet
- [x] Add a research-or-writing adapter that maps sources/notes/objectives into the generic packet
- [x] Add domain capability descriptors so runtimes know which packet fields they can use
- [x] Move canonical trace/replay storage toward generic packet fields rather than coding-only fields
- [ ] Improve candidate artifact ranking within adapters
- [x] Add changed-artifact awareness
- [x] Add symbol/region extraction hooks where the domain supports them
- [x] Improve retrieval of recent failures / prior attempts
- [x] Handle large task contexts more predictably with budgeted packing
- [x] Add adapter-specific artifact exclusion rules (for example build outputs, generated assets, cached artifacts)

## Phase 6 — Advice schema and injection layer
- [x] Define a generic advice schema with domain-neutral fields (`focus_targets`, `recommended_plan`, `avoid`, `likely_failure_modes`, `confidence`, `notes`)
- [x] Make the generic advice schema the canonical stored/rendered form
- [x] Preserve coding-specific convenience fields only as adapter extensions, not core requirements
- [ ] Add advice rendering templates for different executor types (chat model, agent loop, API client, human operator)
- [ ] Add prompt builders that inject advice without assuming a coding workflow
- [ ] Remove coding-first examples and wording from the default runtime prompt unless supplied by a coding adapter
- [x] Add executor-side policy for how advice is prepended, merged, or gated
- [x] Add structured trace capture for exactly what advice was injected into each executor run
- [x] Add calibration guidance for confidence and abstention behavior

## Phase 7 — Evaluation and replay
- [ ] Add golden eval fixtures for each supported domain
- [x] Define fixture schema with frozen generic packet input, expected good guidance targets, and anti-targets
- [x] Add offline steering scorer for artifact targeting, plan quality, failure-mode quality, and noisy-target rate
- [x] Make replay operate on the canonical generic packet, not coding-only trace assumptions
- [x] Add replay harness for stored traces
- [ ] Re-run current advisor versions against historical packets and compare against prior advice and fixture labels
- [ ] Measure no-advisor vs advisor executor behavior on the same tasks with the same black-box model
- [ ] Track comparative metrics beyond basic hit-rate (task success, retries, wall-clock time, token use, dead-end first moves, unnecessary edits, artifact-target rate)
- [x] Add human review rubric for advice usefulness, over-steering, and calibration
- [ ] Add regression-oriented eval docs and locked hard-case suites

## Phase 8 — Reward system and training data pipeline
- [x] Define the reward model inputs from canonical packet/advice/executor/verifier state plus human ratings and trajectory features
- [x] Implement reward computation for both offline replay and live runs
- [x] Add normalized reward components (`task_success`, `efficiency`, `targeting_quality`, `constraint_compliance`, `human_usefulness`)
- [x] Add curation flow for exported traces and reward-labeled examples
- [x] Add quality scoring and filtering before training
- [x] Add split policy (train/val/test) with task-family and repo-family leakage protection
- [x] Add deduping and hard-case bucketing
- [x] Add negative-example capture for harmful or noisy advice
- [x] Version datasets, reward configs, and labeling rules
- [x] Document feedback loop notes for future tuning

## Phase 9 — Advisor optimization loop
- [x] Add training pipeline for advisor-model improvement from reward-labeled data
- [x] Support at least one supervised warm-start path and one preference/reward-optimization path
- [x] Add experiment config for student advisor model, target executor, domain mix, and reward weights
- [x] Add checkpoint evaluation against frozen replay/eval suites
- [x] Add transfer experiments: advisor trained with lower-cost executor context but evaluated against stronger black-box executor
- [x] Add ablations for packet fields, advice schema fields, and reward components
- [x] Add rollback criteria when a newly trained advisor regresses on baseline suites

## Phase 10 — Orchestration and live product loop
- [x] Add a first-class runner for advisor -> executor -> verifier -> reward capture
- [x] Make the canonical lineage generic-packet-first rather than coding-trace-first
- [x] Add pluggable executor interfaces for frontier chat APIs, coding agents, and future domain-specific workers
- [x] Add verifier interfaces for build/test checks, screenshot comparison, rubric graders, and human review
- [x] Persist full run lineage linking packet, advice, executor output, verifier output, and reward
- [x] Add replayable run manifests so experiments can be reproduced exactly
- [x] Add online A/B routing for baseline vs advisor-assisted execution
- [x] Add support for optional second-pass advisor review after executor output

## Phase 11 — Security, privacy, and observability
- [x] Document stored data, reward logs, and retention expectations
- [x] Add redaction and safe-default guidance for packets, traces, and human labels
- [x] Add structured logs and run-id tracing across advisor, executor, verifier, and reward stages
- [x] Add metrics surface or export path for eval and live runs
- [x] Add production operating notes for multi-tenant or hosted deployments
- [x] Add audit notes for dataset provenance and reward-label lineage

## Phase 12 — Real executor and verifier integrations
- [x] Add real frontier-chat, coding-agent, and domain-worker executor integrations
- [x] Add real build/test, rubric, screenshot, and human-review verifier integrations
- [x] Freeze the baseline vs advisor-assisted parity contract across the same executor/verifier path
- [x] Normalize executor/verifier outputs into the canonical lineage without drift

## Phase 13 — Canonical benchmark and replay suite
- [x] Freeze reproducible benchmark splits for core domains
- [x] Add benchmark manifests that capture packet hash, executor config, verifier set, routing arm, and reward version
- [x] Add reproducible baseline vs advisor-assisted results generation
- [x] Add ablation-friendly benchmark reporting

## Phase 14 — Real training loop and checkpoint lifecycle
- [x] Train at least one real advisor checkpoint from repo-generated data
- [x] Evaluate checkpoints against frozen replay/benchmark suites
- [x] Add checkpoint promotion / rollback based on benchmark outcomes
- [x] Persist exact experiment manifests for each training run
- [x] Record profile-local GRPO backend artifacts and profile-owned candidate checkpoints
- [x] Gate profile promotion on profile-local benchmark deltas with explicit rollback reasons

## Phase 15 — Operator surface and deployment path
- [x] Add deployable single-tenant and hosted/service operating paths
- [x] Add run inspection / dashboard surface for lineage, rewards, failures, and benchmark summaries
- [x] Add background jobs, queueing, and resumability for long runs
- [x] Enforce retention / archival / rotation behavior operationally
- [x] Add typed queued train/eval/promote operator jobs without bypassing benchmark-gated promotion

## Phase 16 — Paper-faithful results pass
- [x] Run canonical baseline vs advisor-assisted studies on frozen suites
- [x] Run packet/advice/reward ablations and transfer experiments
- [x] Produce a credible results report with benchmark tables, failure taxonomy, and provenance coverage
- [x] Document remaining divergences from the source paper explicitly

## Phase 17 — Finished product hardening
- [x] Harden auth, tenancy, isolation, and operator runbooks
- [x] Add release gates based on benchmark regression thresholds
- [x] Finalize packaging / backup / import-export / alerting paths
- [x] Lock truth-surface schemas and version benchmark / reward / experiment contracts
