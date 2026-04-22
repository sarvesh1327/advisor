# Phase 7 Evaluation Guide

Phase 7 establishes a generic-first evaluation loop for Advisor.

## Canonical evaluation inputs

Use these as the primary evaluation surface:
- canonical input packet
- canonical advice / injected advice
- injection policy
- execution outcome
- optional human review rubric

Do not build new evaluation logic around coding-only fields when a generic equivalent already exists.

## Fixture rules

Store regression fixtures under:
- `tests/agent/advisor/fixtures/`

Each fixture should contain:
- `fixture_id`
- `domain`
- `description`
- `input_packet`
- `expected_advice`
- `human_review_rubric`

## Human review rubric

Default rubric:
- scale: `0, 1, 2, 3`
- criteria:
  - `helpfulness`
  - `oversteer`
  - `calibration`

Interpretation:
- `0` = harmful
- `1` = weak
- `2` = useful
- `3` = very good

## Replay flow

Replay should prefer:
1. canonical injected advice
2. canonical stored advice
3. canonical input packet

This mirrors what the executor actually saw.

## Generic-first metrics

Minimum metrics to keep:
- `focus_target_hit_rate`
- `injected_advice_rate`
- `success_runs`
- `failure_runs`
- `partial_runs`
- `avg_retries`

Compatibility metrics like `file_hit_rate` may remain, but they are not the only evaluation surface.

## Regression workflow

When a bad run or flaky run is discovered:
1. create or update a fixture in `tests/agent/advisor/fixtures/`
2. add or update a deterministic scorer/replay test
3. rerun:
   - `ruff check .`
   - `python -m pytest tests/agent/advisor -q`

## Review rule

A Phase 7 eval artifact is only useful if it is:
- replayable
- generic-first
- deterministic enough for regression detection
- concise enough to maintain
