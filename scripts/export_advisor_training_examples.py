#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent.advisor.labeling import export_training_examples
from agent.advisor.settings import AdvisorSettings
from agent.advisor.trace_store import AdvisorTraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export advisor training examples to JSONL")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--db", dest="db_path", default=None, help="Override advisor SQLite DB path")
    parser.add_argument("--split", default="train", help="Dataset split label")
    args = parser.parse_args()

    settings = AdvisorSettings.from_env()
    db_path = args.db_path or settings.trace_db_path
    store = AdvisorTraceStore(db_path)
    count = export_training_examples(store, args.output, split=args.split)
    print(f"exported {count} example(s) to {args.output}")


if __name__ == "__main__":
    main()
