.PHONY: venv install install-dev test lint fmt ci

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -e .

install-dev:
	. .venv/bin/activate && pip install -e '.[advisor,dev]'

test:
	. .venv/bin/activate && pytest tests/agent/advisor -q

lint:
	. .venv/bin/activate && ruff check .

fmt:
	. .venv/bin/activate && ruff format .

ci:
	. .venv/bin/activate && ruff check . && pytest tests/agent/advisor -q
