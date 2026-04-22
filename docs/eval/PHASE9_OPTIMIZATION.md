# Phase 9 optimization loop

Phase 9 adds a lightweight but structured optimization scaffold on top of reward-labeled traces.

## What exists now

The optimization loop currently provides:
- experiment config for:
  - student advisor model
  - target executor
  - optional transfer executor
  - domain mix
  - reward preset
  - supervised vs preference mode
- dataset manifest building from reward-labeled traces
- checkpoint comparison against frozen metrics
- transfer-executor reporting
- ablation-plan generation
- rollback decisions from configured thresholds

## Supported training modes

`ExperimentConfig.training_mode` supports:
- `supervised`
- `preference`

`ExperimentConfig.preference_training_mode` currently supports naming the preference path:
- `dpo`
- `simpo`
- `orpo`

This phase is orchestration/config-first. It does not train a model inside this repo yet.

## Core objects

- `ExperimentConfig`
- `RollbackPolicy`
- `AblationSpec`
- `build_dataset_manifest()`
- `evaluate_checkpoint()`
- `generate_ablation_plans()`
- `should_rollback()`

## Dataset manifest intent

`build_dataset_manifest()` turns the trace store into an experiment-ready summary:
- keeps reward-labeled examples only
- respects quality filtering while retaining negative examples
- reports split counts
- reports hard-case buckets
- records the effective reward preset/weights used for the experiment

## Checkpoint evaluation intent

`evaluate_checkpoint()` compares candidate checkpoint metrics to a frozen baseline and returns:
- metric deltas
- optional transfer metrics
- rollback recommendation

The intended source of these metrics is the frozen replay/eval suites from Phase 7.

## Rollback policy

Rollback is triggered when either:
- success delta drops below `min_success_delta`
- score delta drops below `min_score_delta`

This makes rollback rules explicit and reproducible.

## Ablations

`generate_ablation_plans()` currently supports ablations over:
- packet fields
- advice fields
- reward components

This is enough to run controlled sweeps without encoding experiment logic ad hoc.

## What Phase 10 can assume

Phase 10 can now build on:
- stable experiment config
- reward-aware dataset manifests
- explicit checkpoint comparison
- transfer evaluation hooks
- rollback thresholds
- ablation-plan generation
