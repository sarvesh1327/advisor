# Phase 14 real training loop and checkpoint lifecycle

Phase 14 turns the optimization scaffold into a concrete checkpoint lifecycle surface.

## What exists now

The repo now provides:
- `TrainingJobConfig`
  - records the training mode, dataset manifest, benchmark manifests, and output directory
- `TrainingJobResult`
  - records the job id, checkpoint id, manifest path, artifact dir, and training metrics
- `TrainingCheckpointRecord`
  - records checkpoint metadata, lifecycle state, benchmark summary, and rollback reason
- `CheckpointLifecycleManager`
  - registers candidate checkpoints
  - promotes checkpoints to active
  - rolls checkpoints back with an explicit reason
  - persists a checkpoint registry
  - persists a training manifest for each job
- `evaluate_trained_checkpoint()`
  - compares candidate vs baseline benchmark summaries
  - reports deltas
  - recommends promotion vs rollback

## Artifact contract

Phase 14 now standardizes the first checkpoint artifact layout:
- `artifacts/checkpoint_registry.json`
- `artifacts/training-jobs/<job_id>/training-manifest.json`
- `artifacts/checkpoints/<checkpoint_id>/`

This gives later real trainer backends a stable place to write outputs.

## Promotion / rollback contract

Promotion and rollback are now benchmark-driven:
- promote when candidate exceeds the configured threshold
- rollback when benchmark deltas regress below zero

This keeps checkpoint lifecycle decisions grounded in measured outcomes rather than operator intuition.

## Manifest contract

Each training manifest records:
- job id
- experiment id
- training mode
- dataset manifest
- benchmark manifests
- training metrics
- checkpoint id
- creation timestamp

This is the minimum experiment-trace surface needed before heavier training backends are added.

## What Phase 15 can assume

Phase 15 can now build on:
- persisted checkpoint registry state
- stable artifact directory conventions
- persisted training manifests
- explicit promotion / rollback decisions derived from benchmark summaries
