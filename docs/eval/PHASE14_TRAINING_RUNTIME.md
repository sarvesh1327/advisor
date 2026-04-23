# Phase 14 real training loop and checkpoint lifecycle

Phase 14 turns the optimization scaffold into a concrete checkpoint lifecycle surface.

Phase D extends that surface with pre-backend rollout contracts so training jobs can carry replay-friendly rollout artifacts before any GRPO backend is implemented.

## What exists now

The repo now provides:
- `TrainingJobConfig`
  - records the training mode, dataset manifest, benchmark manifests, and output directory
  - records advisor profile id, backend name, rollout-group linkage, and backend artifact paths for profile-local jobs
- `TrainingJobResult`
  - records the job id, checkpoint id, manifest path, artifact dir, training metrics, advisor profile id, backend name, rollout-group id, and backend artifact paths
- `TrainingCheckpointRecord`
  - records checkpoint metadata, lifecycle state, benchmark summary, rollback reason, and advisor profile ownership
- `CheckpointLifecycleManager`
  - registers candidate checkpoints
  - promotes checkpoints to active
  - rolls checkpoints back with an explicit reason
  - persists a checkpoint registry
  - persists a training manifest for each job
  - persists rollout-group manifests for later backend consumption
- `evaluate_trained_checkpoint()`
  - compares candidate vs baseline benchmark summaries
  - reports deltas
  - recommends promotion vs rollback
- `training_rollouts.py`
  - defines single-rollout and grouped-rollout contracts
  - supports both single-turn and multi-turn rollout payloads
  - returns replay-friendly rollout artifacts carrying packet, advice, executor output, verifier outputs, reward label, and diagnostics
- `training_backends.py`
  - defines the profile-local GRPO backend request/result contract
  - consumes rollout-group artifacts and emits deterministic checkpoint/backend manifests
- `run_profile_training_job()`
  - resolves advisor profile training config
  - records rollout-group artifacts for the job
  - runs the GRPO backend
  - registers a candidate checkpoint owned by the advisor profile
  - persists a profile-local training manifest with backend metadata

## Artifact contract

Phase 14 plus the Phase D rollout extension now standardize this artifact layout:
- `artifacts/checkpoint_registry.json`
- `artifacts/training-jobs/<job_id>/training-manifest.json`
- `artifacts/training-jobs/<job_id>/rollout-group.json`
- `artifacts/training-jobs/<job_id>/backend-manifest.json`
- `artifacts/checkpoints/<profile_id>/<checkpoint_id>/`

This gives later real trainer backends a stable place to write outputs and consume rollout artifacts.

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
- advisor profile id
- backend name
- rollout-group id/path
- backend artifact paths
- creation timestamp

Each rollout-group manifest records:
- job id
- group id
- advisor profile id
- rollout count
- reward values
- rollout summary
- serialized rollout results

This is the minimum experiment-trace surface needed before heavier training backends are added.

## What Phase 15 can assume

Phase 15 can now build on:
- persisted checkpoint registry state
- stable artifact directory conventions
- persisted training manifests
- persisted rollout-group manifests
- explicit promotion / rollback decisions derived from benchmark summaries
