#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.metrics import summarize_runs
from agent.advisor.storage.trace_store import AdvisorTraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize advisor run metrics from SQLite trace store")
    parser.add_argument("--db", dest="db_path", default=None, help="Override advisor SQLite DB path")
    args = parser.parse_args()

    settings = AdvisorSettings.from_env()
    db_path = args.db_path or settings.trace_db_path
    store = AdvisorTraceStore(db_path)
    print(json.dumps(summarize_runs(store), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
