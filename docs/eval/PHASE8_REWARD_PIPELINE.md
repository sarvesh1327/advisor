# Phase 8 reward pipeline

Phase 8 turns Advisor traces into reward-labeled training examples.

## Canonical reward inputs

Reward is computed from:
- canonical input packet
- canonical advice block
- executor outcome
- optional human rating
- optional constraint-violation list

The current reward version is `phase8-v1`.

## Normalized reward components

Each run gets five normalized component scores in `[0, 1]`:
- `task_success`
- `efficiency`
- `targeting_quality`
- `constraint_compliance`
- `human_usefulness`

These are combined into a weighted `total_reward` and mirrored into `quality_score` for curation.

## Reward labels in the trace store

Reward labels are persisted per run in the SQLite trace store.

Stored fields include:
- component scores
- total reward
- quality score
- dataset split
- example type (`positive`, `negative`, `neutral`)
- hard-case bucket
- reward version
- notes

## Export and curation rules

`export_training_examples()` now exports only runs with reward labels.

Curation behavior:
- filters out low-quality neutral examples with `min_quality_score`
- keeps negative examples even when their score is low
- assigns a stable split from repo-family + task-family when no explicit split override is given
- dedupes near-identical examples before writing JSONL
- preserves hard-case buckets for later focused replay or training

## Hard-case buckets

Current hard-case buckets are lightweight and deterministic:
- `constraint_failure`
- `failed_execution`
- `targeting_miss`
- `inefficient_execution`

## Feedback-loop notes

Recommended loop:
1. record packet/advice/outcome
2. compute and persist reward label
3. export curated JSONL with quality filtering
4. train or evaluate against the frozen export
5. inspect hard-case buckets and negative examples before changing reward weights

## Split policy

The default split policy hashes:
- repo family
- task family

This is a lightweight leakage guard so closely related examples land in the same split.

## Next phase handoff

Phase 9 can now assume:
- reward-labeled datasets exist
- low-quality neutral traces can be filtered out
- negative examples are preserved
- hard cases can be sampled intentionally
- dataset/reward versions are attached to each exported row
