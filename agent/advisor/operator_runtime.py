from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .benchmark import BenchmarkRunManifest, compare_benchmark_arms
from .observability import export_live_metrics
from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore


class DeploymentProfile(BaseModel):
    mode: Literal["single_tenant", "hosted"]
    bind_host: str
    port: int
    auth_boundary: str
    storage_root: str
    metadata: dict = Field(default_factory=dict)


class OperatorJobRecord(BaseModel):
    job_id: str
    job_type: str
    status: Literal["queued", "running", "completed", "failed"]
    payload: dict = Field(default_factory=dict)
    resume_token: str | None = None
    result: dict = Field(default_factory=dict)
    last_error: str | None = None
    attempts: int = 0
    created_at: str
    updated_at: str


class OperatorJobRequest(BaseModel):
    job_type: str
    payload: dict = Field(default_factory=dict)
    resume_token: str | None = None


class OperatorJobQueue:
    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path).expanduser()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue_job(
        self,
        *,
        job_type: str,
        payload: dict,
        job_id: str | None = None,
        resume_token: str | None = None,
    ) -> OperatorJobRecord:
        now = _utc_now().isoformat()
        record = OperatorJobRecord(
            job_id=job_id or f"job_{uuid.uuid4().hex[:12]}",
            job_type=job_type,
            status="queued",
            payload=payload,
            resume_token=resume_token,
            created_at=now,
            updated_at=now,
        )
        records = [item for item in self.list_jobs() if item.job_id != record.job_id]
        records.append(record)
        self._write_jobs(records)
        return record

    def list_jobs(self) -> list[OperatorJobRecord]:
        if not self.state_path.exists():
            return []
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return [OperatorJobRecord.model_validate(item) for item in payload]

    def update_job(
        self,
        job_id: str,
        *,
        status: Literal["queued", "running", "completed", "failed"],
        result: dict | None = None,
        last_error: str | None = None,
    ) -> OperatorJobRecord:
        records = self.list_jobs()
        updated_record = None
        updated: list[OperatorJobRecord] = []
        for item in records:
            if item.job_id == job_id:
                attempts = item.attempts + (1 if status == "running" else 0)
                updated_record = item.model_copy(
                    update={
                        "status": status,
                        "result": result or item.result,
                        "last_error": last_error,
                        "attempts": attempts,
                        "updated_at": _utc_now().isoformat(),
                    }
                )
                updated.append(updated_record)
            else:
                updated.append(item)
        if updated_record is None:
            raise ValueError(f"unknown job_id: {job_id}")
        self._write_jobs(updated)
        return updated_record

    def resume_incomplete_jobs(self) -> list[OperatorJobRecord]:
        resumed: list[OperatorJobRecord] = []
        updated: list[OperatorJobRecord] = []
        for item in self.list_jobs():
            if item.status in {"running", "failed"} and item.resume_token:
                resumed_record = item.model_copy(
                    update={
                        "status": "queued",
                        "last_error": None,
                        "updated_at": _utc_now().isoformat(),
                    }
                )
                resumed.append(resumed_record)
                updated.append(resumed_record)
            else:
                updated.append(item)
        self._write_jobs(updated)
        return resumed

    def _write_jobs(self, records: list[OperatorJobRecord]) -> None:
        self.state_path.write_text(
            json.dumps([item.model_dump() for item in records], indent=2, sort_keys=True),
            encoding="utf-8",
        )


class RetentionEnforcer:
    def __init__(self, *, store: AdvisorTraceStore, settings: AdvisorSettings, archive_root: str | Path | None = None):
        self.store = store
        self.settings = settings
        self.archive_root = Path(archive_root or Path(settings.trace_db_path).expanduser().parent / "archive")
        self.archive_root.mkdir(parents=True, exist_ok=True)

    def enforce(self, *, now: datetime | None = None) -> dict:
        active_now = now or _utc_now()
        cutoff = active_now - timedelta(days=self.settings.retention_days)
        stale_runs = [run for run in self.store.list_runs(include_context=True) if _is_stale(run.get("started_at"), cutoff)]
        archived_runs = 0
        deleted_runs = 0
        archived_runs_path = None
        if stale_runs:
            archived_runs_path = self.archive_root / f"runs-{active_now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
            with archived_runs_path.open("w", encoding="utf-8") as handle:
                for run in stale_runs:
                    payload = dict(run)
                    payload["lineage"] = self.store.get_lineage(run["run_id"])
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
            archived_runs = len(stale_runs)
            deleted_runs = self.store.delete_runs([run["run_id"] for run in stale_runs])

        event_report = self._rotate_event_log(cutoff=cutoff, active_now=active_now)
        return {
            "retention_days": self.settings.retention_days,
            "cutoff": cutoff.isoformat(),
            "archived_runs": archived_runs,
            "deleted_runs": deleted_runs,
            "archived_runs_path": str(archived_runs_path) if archived_runs_path else None,
            **event_report,
        }

    def _rotate_event_log(self, *, cutoff: datetime, active_now: datetime) -> dict:
        event_path = Path(self.settings.event_log_path).expanduser()
        if not event_path.exists():
            return {
                "archived_event_lines": 0,
                "retained_event_lines": 0,
                "archived_events_path": None,
            }

        lines = event_path.read_text(encoding="utf-8").splitlines()
        stale_lines: list[str] = []
        retained_lines: list[str] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                retained_lines.append(line)
                continue
            if _is_stale(event.get("ts"), cutoff):
                stale_lines.append(json.dumps(event, sort_keys=True))
            else:
                retained_lines.append(json.dumps(event, sort_keys=True))

        archived_events_path = None
        if stale_lines:
            archived_events_path = self.archive_root / f"events-{active_now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
            archived_events_path.write_text("\n".join(stale_lines) + "\n", encoding="utf-8")
        event_path.write_text(("\n".join(retained_lines) + "\n") if retained_lines else "", encoding="utf-8")
        return {
            "archived_event_lines": len(stale_lines),
            "retained_event_lines": len(retained_lines),
            "archived_events_path": str(archived_events_path) if archived_events_path else None,
        }


def build_deployment_profile(
    *,
    settings: AdvisorSettings,
    mode: Literal["single_tenant", "hosted"],
    bind_host: str | None = None,
    port: int | None = None,
) -> DeploymentProfile:
    storage_root = str(Path(settings.trace_db_path).expanduser().parent)
    if mode == "hosted":
        return DeploymentProfile(
            mode=mode,
            bind_host=bind_host or "0.0.0.0",
            port=port or 8080,
            auth_boundary="external auth proxy required",
            storage_root=storage_root,
            metadata={
                "hosted_mode": True,
                "trace_db_path": settings.trace_db_path,
                "event_log_path": settings.event_log_path,
            },
        )
    return DeploymentProfile(
        mode=mode,
        bind_host=bind_host or "127.0.0.1",
        port=port or 8000,
        auth_boundary="local process boundary",
        storage_root=storage_root,
        metadata={
            "hosted_mode": settings.hosted_mode,
            "trace_db_path": settings.trace_db_path,
            "event_log_path": settings.event_log_path,
        },
    )


def build_operator_snapshot(
    *,
    store: AdvisorTraceStore,
    settings: AdvisorSettings,
    deployment: DeploymentProfile,
    benchmark_manifests: list[BenchmarkRunManifest] | None = None,
    job_records: list[OperatorJobRecord] | list[dict] | None = None,
    limit: int = 10,
) -> dict:
    runs = store.list_runs(include_context=False)[:limit]
    metrics = export_live_metrics(store)
    benchmark_summary = compare_benchmark_arms(benchmark_manifests or []) if benchmark_manifests else {}
    normalized_jobs = [item.model_dump() if isinstance(item, OperatorJobRecord) else item for item in (job_records or [])]
    run_summaries = []
    for row in runs:
        lineage = store.get_lineage(row["run_id"])
        reward_label = row.get("reward_label") or {}
        run_summaries.append(
            {
                "run_id": row["run_id"],
                "started_at": row.get("started_at"),
                "task_text": row["task_text"],
                "task_type": row["task_type"],
                "status": ((row.get("outcome") or {}).get("status")) or "pending",
                "routing_arm": ((lineage or {}).get("manifest") or {}).get("routing_decision", {}).get("arm"),
                "reward_total": reward_label.get("total_reward"),
                "reward_version": reward_label.get("reward_version"),
                "lineage_available": lineage is not None,
                "provenance": {
                    "run_id": row["run_id"],
                    "reward_label_available": bool(row.get("reward_label")),
                    "lineage_available": lineage is not None,
                },
            }
        )
    return {
        "deployment": deployment.model_dump(),
        "live_metrics": metrics.model_dump(),
        "runs": run_summaries,
        "benchmark_summary": benchmark_summary,
        "jobs": normalized_jobs,
        "retention": {
            "retention_days": settings.retention_days,
            "archive_root": str(Path(settings.trace_db_path).expanduser().parent / "archive"),
        },
    }


def _is_stale(timestamp: str | None, cutoff: datetime) -> bool:
    parsed = _parse_timestamp(timestamp)
    return parsed is not None and parsed < cutoff


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)
