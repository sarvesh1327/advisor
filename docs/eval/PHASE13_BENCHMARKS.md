# Phase 13 canonical benchmark and replay suite

Phase 13 turns the replay/eval substrate into a reproducible benchmark surface.

## What exists now

The repo now provides:
- `freeze_benchmark_suite()`
  - deterministically assigns fixtures into:
    - `train_pool`
    - `validation`
    - `test`
- `BenchmarkCase`
- `BenchmarkSuite`
- `BenchmarkRunManifest`
- `build_benchmark_run_manifest()`
  - binds a run to a fixture and split
  - records canonical benchmark metadata
- `compare_benchmark_arms()`
  - aggregates baseline vs advisor-assisted benchmark results
  - produces deterministic summaries and deltas
  - exposes ablation axes for later controlled sweeps

## Frozen split contract

Frozen suite assignment is deterministic from:
- suite id
- fixture domain
- fixture id

That means the same fixture set always produces the same benchmark split layout.

## Benchmark manifest contract

Each benchmark run manifest records:
- `run_id`
- `fixture_id`
- `domain`
- `split`
- `packet_hash`
- `executor_config`
- `verifier_set`
- `routing_arm`
- `reward_version`
- `score`

This is the minimum truth surface needed for reproducible benchmark reporting.

## Baseline vs advisor reporting

`compare_benchmark_arms()` currently reports:
- arm-level summaries
- split-level summaries
- domain-level summaries
- advisor-minus-baseline deltas
- ablation axes across:
  - domains
  - executor kinds
  - reward versions
  - splits
  - verifier sets

This is enough to start real benchmark tables and preserve comparability.

## What Phase 14 can assume

Phase 14 can now build on:
- frozen benchmark splits
- reproducible benchmark run manifests
- deterministic baseline vs advisor-assisted summaries
- ablation-friendly benchmark metadata ready for checkpoint evaluation
