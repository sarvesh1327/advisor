from __future__ import annotations

from .trace_store import AdvisorTraceStore


def summarize_runs(store: AdvisorTraceStore) -> dict:
    runs = store.list_runs()
    total = len(runs)
    success = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "success")
    failure = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "failure")
    partial = sum(1 for row in runs if (row.get("outcome") or {}).get("status") == "partial")

    file_hit_scores = []
    focus_hit_scores = []
    retry_values = []
    injected_runs = 0
    for row in runs:
        outcome = row.get("outcome") or {}
        advice = row.get("advice") or {}
        touched = set(outcome.get("files_touched") or [])
        advised_files = {item.get("path") for item in advice.get("relevant_files") or [] if item.get("path")}
        if advised_files:
            # This is a coarse compatibility proxy: did file-specific advice mention any touched file?
            file_hit_scores.append(1.0 if touched & advised_files else 0.0)
        focus_targets = {item.get("locator") for item in advice.get("focus_targets") or [] if item.get("locator")}
        if focus_targets:
            # Generic-first metric: did any advised focus target match an executed artifact?
            focus_hit_scores.append(1.0 if touched & focus_targets else 0.0)
        if row.get("injected_rendered_advice"):
            injected_runs += 1
        if outcome:
            retry_values.append(int(outcome.get("retries") or 0))

    file_hit_rate = (sum(file_hit_scores) / len(file_hit_scores)) if file_hit_scores else 0.0
    focus_target_hit_rate = (sum(focus_hit_scores) / len(focus_hit_scores)) if focus_hit_scores else 0.0
    avg_retries = (sum(retry_values) / len(retry_values)) if retry_values else 0.0
    injected_advice_rate = (injected_runs / total) if total else 0.0
    return {
        "total_runs": total,
        "success_runs": success,
        "failure_runs": failure,
        "partial_runs": partial,
        "file_hit_rate": file_hit_rate,
        "focus_target_hit_rate": focus_target_hit_rate,
        "injected_advice_rate": injected_advice_rate,
        "avg_retries": avg_retries,
    }
