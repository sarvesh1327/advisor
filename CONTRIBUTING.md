# Contributing to Advisor

Thanks for contributing to Advisor.

## Development environment

Create and activate a local virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install development dependencies:

```bash
pip install -e '.[advisor,dev]'
```

## Core checks

Run the focused test suite before opening a PR:

```bash
pytest tests/agent/advisor -q
```

Run lint checks:

```bash
ruff check .
```

## Workflow

- keep changes small and reviewable
- add or update tests first when changing behavior
- prefer stable public interfaces over incidental internals
- update README/docs when public usage changes
- treat `docs/PAPER_FOUNDATION.md` as a design constraint, not optional reading
- prefer generic advisor abstractions in the core and isolate coding-only logic behind adapters or extension fields
- treat generic packet/advice/replay state as canonical whenever a generic equivalent exists
- when adding new evaluation or execution paths, make sure they preserve replayability and produce reward-usable signals

## Pull requests

Please include:
- what changed
- why it changed
- how it was tested
- how it supports or intentionally scopes around the paper-aligned roadmap
- any follow-up work left out of the PR
