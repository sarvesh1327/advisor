# Phase 16 — Paper-faithful results pass

## Goal
Produce the first credible, audit-friendly evidence report from Advisor’s frozen benchmark and live lineage surfaces.

## What was added

### Results-pass module
- `agent/advisor/results_pass.py`
- canonical study summarization from frozen benchmark manifests
- ablation summarization for packet/advice/reward variants
- transfer summarization across target and transfer executors
- failure taxonomy generation from stored run outcomes
- provenance coverage reporting from lineage + reward labels
- explicit default divergence list against the source paper
- final report builder and JSON writer

### Public exports
- updated `agent/advisor/__init__.py`
- exported:
  - `summarize_canonical_study(...)`
  - `summarize_ablation_results(...)`
  - `summarize_transfer_results(...)`
  - `build_failure_taxonomy(...)`
  - `summarize_provenance_coverage(...)`
  - `default_paper_divergences()`
  - `build_phase16_results_report(...)`
  - `write_phase16_results_report(...)`

## What Phase 16 now covers
- canonical baseline vs advisor study summaries on frozen benchmark manifests
- ablation result rollups across:
  - packet fields
  - advice fields
  - reward components
- transfer executor reporting with explicit executor-pair rows
- failure taxonomy from stored outcomes for reportable failure analysis
- provenance coverage for:
  - lineage presence
  - reward-label presence
  - joint audit coverage
- explicit documented divergences from the paper so the repo does not overclaim fidelity

## Evidence/report shape
The composed Phase 16 report now includes:
- `canonical_study`
- `ablation_results`
- `transfer_results`
- `failure_taxonomy`
- `provenance_coverage`
- `paper_divergences`

## Verification

### Red
- `python -m pytest tests/agent/advisor/test_results_pass.py -q`
- failed first with:
  - `ModuleNotFoundError: agent.advisor.results_pass`

### Green targeted
- `python -m pytest tests/agent/advisor/test_results_pass.py tests/agent/advisor/test_benchmark.py tests/agent/advisor/test_training_pipeline.py -q`
- passed

### Lint
- `ruff check .`
- passed

### Full suite
- `python -m pytest tests/agent/advisor -q`
- passed

## Done criteria coverage
- canonical baseline vs advisor study on frozen suites: yes
- packet/advice/reward ablations and transfer experiments: yes
- credible results report with benchmark tables/failure taxonomy/provenance: yes
- remaining paper divergences documented explicitly: yes
