# Phase 5 — Measurement over time

## Goal
Add one reusable report surface that answers how each advisor profile changes over time across repeated training/eval/promote cycles.

## Landed surface
- `agent.advisor.build_phase5_measurement_report(...)`
- implementation: `agent/advisor/evaluation/measurement.py`

## What the report includes
- per-profile summaries
- checkpoint lineage history
- active checkpoint tracking
- cycle trend history from completed `train-profile`, `eval-profile`, and `promote-checkpoint` jobs
- eval deltas for `overall_score` and `focus_target_recall`
- promotion thresholds and promote/rollback decisions
- artifact fingerprints and change-vs-previous signals for checkpoint evolution

## Bounded guarantees from this phase
- measurement is profile-local
- repeated cycles can be inspected as ordered trend rows
- checkpoint history exposes whether artifacts actually changed over time
- promotion outcomes are visible as trend events, not only hidden in one-off job results

## Verification
- targeted: `python -m pytest tests/agent/advisor/test_measurement.py -q`
- repo-wide: `ruff check .`
- full advisor suite: `python -m pytest tests/agent/advisor -q`

## Not implied yet
- Phase 6 data-quality/reward-sanity hardening is not complete from this step alone
