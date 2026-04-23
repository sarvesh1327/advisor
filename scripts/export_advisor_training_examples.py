#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.storage.labeling import export_training_examples
from agent.advisor.storage.trace_store import AdvisorTraceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export advisor training examples to JSONL")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--db", dest="db_path", default=None, help="Override advisor SQLite DB path")
    parser.add_argument("--split", default=None, help="Optional dataset split override")
    parser.add_argument("--min-quality-score", type=float, default=0.0, help="Filter out low-quality neutral examples")
    args = parser.parse_args()

    settings = AdvisorSettings.from_env()
    db_path = args.db_path or settings.trace_db_path
    store = AdvisorTraceStore(db_path)
    count = export_training_examples(
        store,
        args.output,
        split=args.split,
        min_quality_score=args.min_quality_score,
    )
    print(f"exported {count} example(s) to {args.output}")


if __name__ == "__main__":
    main()
