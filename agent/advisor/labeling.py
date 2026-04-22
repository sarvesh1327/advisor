from __future__ import annotations

import json
from pathlib import Path

from .trace_store import AdvisorTraceStore


def export_training_examples(store: AdvisorTraceStore, output_path: str | Path, split: str = "train") -> int:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for row in store.list_runs(include_context=True):
            outcome = row.get("outcome") or {}
            # Successful runs are the current high-precision source for supervised advice targets.
            if outcome.get("status") != "success":
                continue
            payload = {
                "run_id": row["run_id"],
                "split": split,
                "input": row.get("input") or {},
                "target_advice": row.get("advice") or {},
                "outcome": outcome,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count
