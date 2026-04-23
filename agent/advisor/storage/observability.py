from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.storage.trace_store import AdvisorTraceStore

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(
    r"\b(?:token|secret|apikey|api_key|password)(?:\s+[A-Za-z0-9._-]+)?(?:[\s:=]+)(?:[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)
_ID_FIELDS = {"session_id", "task_id", "conversation_id", "user_id", "tenant_id"}


class LiveMetricsSnapshot(BaseModel):
    total_runs: int = 0
    lineage_runs: int = 0
    reward_labeled_runs: int = 0
    success_runs: int = 0
    failure_runs: int = 0
    partial_runs: int = 0
    arm_counts: dict[str, int] = Field(default_factory=dict)
    verifier_status_counts: dict[str, int] = Field(default_factory=dict)


class RunEventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, *, run_id: str, stage: str, payload: dict | None = None) -> None:
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "stage": stage,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def redact_packet(packet) -> dict:
    payload = packet.model_dump() if hasattr(packet, "model_dump") else dict(packet)
    return _redact_value(payload, parent_key=None)


def export_live_metrics(store: AdvisorTraceStore) -> LiveMetricsSnapshot:
    runs = store.list_runs(include_context=False)
    arm_counts: dict[str, int] = {}
    verifier_status_counts: dict[str, int] = {}
    lineage_runs = 0
    reward_labeled_runs = 0
    success_runs = 0
    failure_runs = 0
    partial_runs = 0

    for row in runs:
        if row.get("reward_label"):
            reward_labeled_runs += 1
        lineage = store.get_lineage(row["run_id"])
        if lineage:
            lineage_runs += 1
            arm = ((lineage.get("manifest") or {}).get("routing_decision") or {}).get("arm") or "unknown"
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            for item in (lineage.get("lineage") or {}).get("verifier_results") or []:
                status = ((item.get("result") or {}).get("status")) or "unknown"
                verifier_status_counts[status] = verifier_status_counts.get(status, 0) + 1
        status = ((row.get("outcome") or {}).get("status")) or "unknown"
        if status == "success":
            success_runs += 1
        elif status == "failure":
            failure_runs += 1
        elif status == "partial":
            partial_runs += 1

    return LiveMetricsSnapshot(
        total_runs=len(runs),
        lineage_runs=lineage_runs,
        reward_labeled_runs=reward_labeled_runs,
        success_runs=success_runs,
        failure_runs=failure_runs,
        partial_runs=partial_runs,
        arm_counts=arm_counts,
        verifier_status_counts=verifier_status_counts,
    )


def build_audit_report(store: AdvisorTraceStore, settings: AdvisorSettings) -> dict:
    metrics = export_live_metrics(store)
    return {
        "stored_data": {
            "trace_db_path": settings.trace_db_path,
            "event_log_path": settings.event_log_path,
            "captures": [
                "packets",
                "advice records",
                "run outcomes",
                "reward labels",
                "run lineage manifests",
                "structured run events",
            ],
        },
        "retention": {
            "retention_days": settings.retention_days,
            "hosted_mode": settings.hosted_mode,
            "expectation": "rotate or archive trace DB and event logs on the configured retention boundary",
        },
        "redaction": {
            "enabled": settings.redact_sensitive_fields,
            "safe_defaults": [
                "emails are redacted",
                "secret/token/password fragments are redacted",
                "session/task/user identifiers are redacted in exported packets",
            ],
        },
        "dataset_provenance": {
            "reward_labeled_runs": metrics.reward_labeled_runs,
            "reward_labels_linked_to_lineage": metrics.lineage_runs,
        },
        "lineage": {
            "lineage_runs": metrics.lineage_runs,
            "arm_counts": metrics.arm_counts,
        },
        "metrics": metrics.model_dump(),
    }


def _redact_value(value, *, parent_key: str | None):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in _ID_FIELDS and isinstance(item, str) and item:
                redacted[key] = "[REDACTED:id]"
            else:
                redacted[key] = _redact_value(item, parent_key=key)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        value = _EMAIL_PATTERN.sub("[REDACTED:email]", value)
        value = _TOKEN_PATTERN.sub(lambda match: _redact_secret_fragment(match.group(0)), value)
        return value
    return value


def _redact_secret_fragment(text: str) -> str:
    match = re.match(r"(?i)(token|secret|apikey|api_key|password)([\s:=]+)(.+)", text)
    if not match:
        return "[REDACTED:secret]"
    key, separator = match.group(1), match.group(2)
    return f"{key}{separator}[REDACTED:secret]"
