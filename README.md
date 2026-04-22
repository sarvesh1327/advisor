# Advisor

Advisor is a standalone extraction of the local pre-execution steering module that was originally embedded inside Hermes.

It does 4 things:
- builds a compact context packet for a task
- runs a local MLX model to produce structured steering advice
- validates and stores advice/output traces in SQLite
- exports basic metrics and training examples

This repo contains the advisor module itself, not the full Hermes runtime.

## Repo layout

- `agent/advisor/` — core advisor code
- `scripts/` — metrics/export helpers
- `tests/agent/advisor/` — focused tests
- `docs/` — existing architecture diagrams and images

## Core modules

- `schemas.py` — pydantic contracts for requests, advice, outcomes
- `settings.py` — local config/env loading
- `context_builder.py` — repo slice, candidate files, task typing, failure lookup
- `runtime_mlx.py` — MLX/MLX-LM inference wrapper
- `validator.py` — trims/dedupes model output into safe structured advice
- `gateway.py` — main entrypoint + optional FastAPI app factory
- `trace_store.py` — SQLite persistence for runs, outcomes, patterns
- `metrics.py` — summary metrics over stored runs
- `labeling.py` — JSONL training export helpers
- `injector.py` — renders advice into an injected hint block

## What this repo includes

- local advisor runtime
- trace store
- export scripts
- focused unit tests
- architecture docs copied from the earlier design work

## What this repo does not include

- full Hermes conversation loop
- the original `run_agent.py` integration
- CLI/product wiring
- complete standalone packaging/runtime polish

## Quick start

Python 3.11+

Create and activate a local virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install base deps:

```bash
pip install -e .
```

Install advisor runtime + dev deps:

```bash
pip install -e '.[advisor,dev]'
```

Run tests:

```bash
pytest tests/agent/advisor -q
```

Current repo status:
- standalone package install works
- focused advisor test suite passes (`14 passed`)
- no Hermes-specific Python references remain in this repo

## Helper scripts

Export successful runs as JSONL:

```bash
python scripts/export_advisor_training_examples.py ./out/train.jsonl
```

Summarize trace metrics:

```bash
python scripts/advisor_metrics_summary.py
```

## Architecture docs

See `docs/ARCHITECTURE.md` for the image index and diagram notes.

## Reference

- arXiv: https://arxiv.org/pdf/2510.02453

## License

Apache License 2.0. See `LICENSE`.

## Origin

Extracted from an earlier Hermes-based prototype, then cleaned into a standalone product-shaped repo.

Notes:
- the original Hermes-specific hook points used to live in `run_agent.py`
- this extracted repo no longer depends on Hermes-specific Python modules
