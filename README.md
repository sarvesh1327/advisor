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

Check the installed CLI:

```bash
advisor version
advisor serve --host 127.0.0.1 --port 8000
```

Current repo status:
- standalone package install works
- focused advisor test suite passes
- no Hermes-specific Python references remain in this repo
- config can now load from `ADVISOR_CONFIG=/path/to/advisor.toml`
- health checks now expose runtime/config state on `/healthz`
- CI workflow now runs lint-only on Python 3.12
- Ruff linting is configured and passes locally
- inference runtime now supports retries, timeout handling, warm-load, and fallback behavior
- reward weights now support named config presets (`balanced`, `conservative`, `human-first`) plus explicit overrides
- Phase 10 orchestration now supports executor/verifier plug-ins, deterministic A/B routing, replayable manifests, and optional second-pass review
- Phase 11 adds redacted packet exports, structured run-event logs, live metrics export, and audit reporting
- GitHub CI installs `.[dev]` only, since MLX runtime deps are Apple-specific and not required for the test suite

## Contributing

See `CONTRIBUTING.md` for local setup, test, and lint workflow.

## Helper scripts

Export successful runs as JSONL:

```bash
python scripts/export_advisor_training_examples.py ./out/train.jsonl --min-quality-score 0.5
```

Summarize trace metrics:

```bash
python scripts/advisor_metrics_summary.py
```

## Architecture docs

See `docs/ARCHITECTURE.md` for the image index and diagram notes.

## Production roadmap

See `docs/PRODUCTION_CHECKLIST.md` for the staged production roadmap toward a generic, paper-faithful advisor implementation with reward-driven improvement.

## Paper foundation

This repo is grounded in How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models.

See `docs/PAPER_FOUNDATION.md` for the repo-level design rules derived from that paper.

## Reference

- How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models
- arXiv abstract: https://arxiv.org/abs/2510.02453
- PDF: https://arxiv.org/pdf/2510.02453

## License

Apache License 2.0. See `LICENSE`.

