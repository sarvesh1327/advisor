# Phase 10 orchestration and live product loop

Phase 10 adds the first live orchestration layer on top of the packet/advice/reward pipeline.

## What exists now

The repo now provides:
- `AdvisorOrchestrator` as a first-class runner for:
  - advisor generation
  - deterministic baseline vs advisor-assisted routing
  - executor dispatch
  - verifier execution
  - outcome creation
  - reward capture
  - lineage persistence
- pluggable executor wrappers for:
  - frontier chat APIs
  - coding agents
  - future domain-specific workers
- pluggable verifier wrappers for:
  - build/test checks
  - screenshot comparison
  - rubric grading
  - human review
- optional second-pass advisor review after executor output
- replayable run manifests persisted to the trace store
- full lineage persistence rooted in the generic `AdvisorInputPacket`

## Core objects

- `AdvisorOrchestrator`
- `DeterministicABRouter`
- `ExecutorDescriptor`
- `VerifierDescriptor`
- `ExecutorRequest`
- `ExecutorRunResult`
- `VerifierResult`
- `RunManifest`
- `RunLineage`
- `LiveRunResult`

## Routing behavior

`DeterministicABRouter` assigns each run to either:
- `baseline`
- `advisor`

The route is deterministic from session/task/run identity so online A/B comparisons stay sticky and reproducible.

## Manifest intent

`RunManifest` captures the replay contract for a live run:
- selected routing arm
- executor identity
- verifier set
- whether review mode was enabled
- replay inputs such as packet hash, system prompt, and advisor model

This gives later experiments an exact description of how a run was executed.

## Lineage intent

`RunLineage` captures the full run chain:
- generic input packet
- primary advice
- optional review advice
- executor result
- verifier results
- outcome
- reward label

This keeps the lineage generic-packet-first and avoids coupling the product to coding-only traces.

## Trace-store additions

The trace store now persists:
- `run_lineages.manifest_json`
- `run_lineages.lineage_json`

Use `get_lineage(run_id)` to recover the exact manifest + lineage pair for replay or auditing.

## What Phase 11 can assume

Phase 11 can now build on:
- a stable live-run orchestration contract
- persistent lineage records for each routed execution
- deterministic baseline/advisor routing decisions
- generic executor/verifier abstractions
- replayable run metadata suitable for observability and audit work
