from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .trace_store import AdvisorTraceStore


def export_training_examples(
    store: AdvisorTraceStore,
    output_path: str | Path,
    split: str | None = None,
    *,
    min_quality_score: float = 0.0,
    advisor_profile_id: str | None = None,
) -> int:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_signatures: set[str] = set()
    runs = sorted(
        store.list_runs(include_context=True),
        key=lambda row: float(((row.get("reward_label") or {}).get("quality_score") or 0.0)),
        reverse=True,
    )
    with output.open("w", encoding="utf-8") as fh:
        for row in runs:
            outcome = row.get("outcome") or {}
            reward_label = row.get("reward_label") or {}
            if outcome.get("status") is None or not reward_label:
                continue
            resolved_profile_id = reward_label.get("advisor_profile_id") or row.get("advisor_profile_id")
            if advisor_profile_id and resolved_profile_id != advisor_profile_id:
                continue
            quality_score = float(reward_label.get("quality_score") or 0.0)
            example_type = reward_label.get("example_type") or "neutral"
            if quality_score < min_quality_score and example_type != "negative":
                continue
            payload = {
                "run_id": row["run_id"],
                "advisor_profile_id": resolved_profile_id,
                "split": split or reward_label.get("dataset_split") or _assign_export_split(row),
                "input": row.get("input") or {},
                "target_advice": row.get("advice") or {},
                "outcome": outcome,
                "reward_label": reward_label,
                "quality_score": quality_score,
                "example_type": example_type,
                "hard_case_bucket": reward_label.get("hard_case_bucket"),
                "dataset_version": reward_label.get("reward_version") or "phase8-v1",
            }
            signature = _example_signature(payload)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def _assign_export_split(row: dict) -> str:
    repo_path = row.get("repo_path") or ""
    task_type = row.get("task_type") or "unknown"
    digest = hashlib.sha256(f"{repo_path}|{task_type}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket <= 2:
        return "val"
    return "train"


def _example_signature(payload: dict) -> str:
    input_payload = payload.get("input") or {}
    target_advice = payload.get("target_advice") or {}
    focus_targets = sorted(item.get("locator") for item in target_advice.get("focus_targets") or [] if item.get("locator"))
    return json.dumps(
        {
            "task_text": input_payload.get("task_text"),
            "repo_path": input_payload.get("repo", {}).get("path"),
            "task_type": input_payload.get("task_type"),
            "focus_targets": focus_targets,
            "example_type": payload.get("example_type"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
