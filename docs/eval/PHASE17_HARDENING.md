# Phase 17 — Finished product hardening

## Goal
Finish the product hardening layer so releases, hosted deployments, operator recovery, and truth-surface versioning are all explicit and testable.

## What was added

### Hardening module
- `agent/advisor/hardening.py`
- release-gate policy and evaluation
- deployment hardening profile generation
- backup/export/import bundle paths
- alert summary generation for failed release gates
- truth-surface contract locking with explicit versioned manifests

### What Phase 17 now covers
- auth / tenancy / isolation guidance for:
  - single-tenant mode
  - hosted mode
- operator runbook lists for deployment/upgrade/recovery flows
- measurable release gates based on:
  - advisor lift
  - reward-label coverage
  - lineage coverage
  - open paper-divergence count
- finalized product-state bundle export/import flow
- locked truth-surface contract versions for:
  - packet schema
  - advice schema
  - benchmark contract
  - reward contract
  - experiment report contract

### CLI surface
Added commands:
- `advisor hardening-profile --mode hosted`
- `advisor release-gate --report-path <report.json>`
- `advisor export-bundle --output-dir <dir>`
- `advisor import-bundle --bundle-path <dir> --target-root <dir>`

## Verification

### Red
- `source .venv/bin/activate && python -m pytest tests/agent/advisor/test_hardening.py -q`
- failed first with:
  - `ModuleNotFoundError: agent.advisor.hardening`

### Green targeted
- `source .venv/bin/activate && python -m pytest tests/agent/advisor/test_hardening.py tests/agent/advisor/test_results_pass.py tests/agent/advisor/test_operator_runtime.py tests/agent/advisor/test_cli.py -q`
- `18 passed`

### Lint
- `source .venv/bin/activate && ruff check .`
- passed

### Full suite
- `source .venv/bin/activate && python -m pytest tests/agent/advisor -q`
- `106 passed`

## Finished-product status
Relative to the accepted roadmap in this repo:
- yes, this completes the final listed phase
- the product is now "finished" against that roadmap
- it is still not a perfect reproduction of the source paper; Phase 16 explicitly documents remaining divergences
