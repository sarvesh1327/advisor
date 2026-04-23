# Phase 6 — Data quality, reward sanity, and anti-regression hardening

## Goal
Add one reusable hardening surface that validates rollout-group training inputs and prevents malformed or regressive promotion evidence from silently advancing checkpoints.

## Landed surface
- `agent.advisor.build_phase6_hardening_report(...)`
- implementation: `agent/advisor/training/hardening.py`

## What Phase 6 now enforces
- rollout-group profile consistency
- reward-label / `reward_values` alignment
- finite bounded reward values in `[0.0, 1.0]`
- duplicate packet/advice training leakage detection using stable profile-local signatures
- flat reward distributions that provide no relative learning signal
- blocked continuous cycles before training when rollout hardening finds blocking issues
- blocked non-promotion fallback for malformed or regressive promotion evidence

## Bounded guarantees from this phase
- invalid rollout groups do not enter the continuous training loop
- duplicate content leakage inside one profile-local rollout group is surfaced deterministically
- promote jobs do not advance checkpoints on rollback-marked or malformed eval evidence
- promotion blockers return structured blocked results instead of silently promoting

## Verification
- targeted: `python -m pytest tests/agent/advisor/test_hardening.py -q`
- repo-wide: `ruff check .`
- full advisor suite: `python -m pytest tests/agent/advisor -q`

## Not implied yet
- Phase 7 operator-facing inspection/control commands are not complete from this step alone
