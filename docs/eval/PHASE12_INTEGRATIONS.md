# Phase 12 real executor and verifier integrations

Phase 12 turns the Phase 10 orchestration contracts into real integration surfaces while preserving baseline vs advisor-assisted comparability.

## What exists now

The repo now provides real integration adapters for:

### Executors
- `FrontierHTTPExecutor`
  - sends the canonical execution payload over HTTP POST
  - normalizes JSON responses into `ExecutorRunResult`
- `CodingAgentSubprocessExecutor`
  - runs a coding-agent style subprocess
  - sends the canonical execution payload over stdin
  - normalizes JSON stdout into `ExecutorRunResult`
- `DomainWorkerSubprocessExecutor`
  - same transport pattern for non-coding workers

### Verifiers
- `BuildTestCommandVerifier`
  - runs build/test commands inside the repo path
  - normalizes pass/fail into `VerifierResult`
- `RubricTextVerifier`
  - checks executor output against required rubric phrases
- `ScreenshotHashVerifier`
  - compares expected vs actual artifacts with SHA-256 hashes
- `HumanReviewFileVerifier`
  - records human review verdicts from a structured JSON file

### Registry
- `IntegrationRegistry`
  - builds executors and verifiers from config dictionaries
  - lets deployments switch integrations without code edits

## Parity contract

Phase 12 preserves the paper-faithful evaluation rule:
- baseline run:
  - same executor
  - same verifier set
  - no injected advisor hint
- advisor-assisted run:
  - same executor
  - same verifier set
  - only the canonical injected advice differs

The integration tests explicitly verify that the manifest parity holds and that only advice injection changes what the executor sees.

## Canonical payload

All real executors receive the same top-level payload:
- `run_id`
- `packet`
- `advice`
- `rendered_advice`
- `routing_decision`

This freezes the request surface before richer convenience fields are added later.

## What Phase 13 can assume

Phase 13 can now build on:
- real executor integrations
- real verifier integrations
- config-driven integration construction
- parity-tested baseline vs advisor-assisted execution
- normalized executor/verifier outputs entering lineage unchanged
