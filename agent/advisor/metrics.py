from __future__ import annotations

from .trace_store import AdvisorTraceStore


def summarize_runs(store: AdvisorTraceStore) -> dict:
    runs = store.list_runs()
    total = len(runs)
    success = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "success")
    failure = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "failure")
    partial = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "partial")

    hit_scores = []
    retry_values = []
    for row in runs:
        outcome = row.get("outcome") or {}
        advice = row.get("advice") or {}
        touched = set(outcome.get("files_touched") or [])
        advised = {item.get("path") for item in advice.get("relevant_files") or [] if item.get("path")}
        if advised:
            # This is a coarse offline proxy: did advice mention any file that execution touched?
            hit_scores.append(1.0 if touched & advised else 0.0)
        if outcome:
            retry_values.append(int(outcome.get("retries") or 0))

    file_hit_rate = (sum(hit_scores) / len(hit_scores)) if hit_scores else 0.0
    avg_retries = (sum(retry_values) / len(retry_values)) if retry_values else 0.0
    return {
        "total_runs": total,
        "success_runs": success,
        "failure_runs": failure,
        "partial_runs": partial,
        "file_hit_rate": file_hit_rate,
        "avg_retries": avg_retries,
    }
