from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.benchmark import BenchmarkRunManifest, compare_benchmark_arms
from agent.advisor.profiles import AdvisorProfileRegistry
from agent.advisor.storage.observability import export_live_metrics
from agent.advisor.storage.trace_store import AdvisorTraceStore
from agent.advisor.training.training_rollouts import TrainingRolloutGroupResult
from agent.advisor.training.training_runtime import (
    CheckpointLifecycleManager,
    evaluate_profile_checkpoint_for_promotion,
    run_profile_training_job,
)

SUPPORTED_OPERATOR_JOB_TYPES = {"train-profile", "eval-profile", "promote-checkpoint"}


class ContinuousTrainingCycleResult(BaseModel):
    train_job: dict
    eval_job: dict
    promote_job: dict | None = None
    promoted: bool = False


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


class TrainProfileJobPayload(BaseModel):
    experiment_id: str
    advisor_profile_id: str
    rollout_group: dict
    benchmark_manifests: list[dict] = Field(default_factory=list)


class EvalProfileJobPayload(BaseModel):
    advisor_profile_id: str
    candidate_checkpoint_id: str
    benchmark_manifests: list[dict] = Field(default_factory=list)
    promotion_threshold: float = 0.05


class PromoteCheckpointJobPayload(BaseModel):
    advisor_profile_id: str
    candidate_checkpoint_id: str
    evaluation: dict = Field(default_factory=dict)


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
        validated_payload = _validate_operator_job_payload(job_type, payload)
        now = _utc_now().isoformat()
        record = OperatorJobRecord(
            job_id=job_id or f"job_{uuid.uuid4().hex[:12]}",
            job_type=job_type,
            status="queued",
            payload=validated_payload.model_dump(),
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


def run_operator_job(
    queue: OperatorJobQueue,
    job_id: str,
    *,
    settings: AdvisorSettings | None = None,
    profile_registry: AdvisorProfileRegistry | None = None,
    lifecycle_manager: CheckpointLifecycleManager | None = None,
    train_profile_fn: Any | None = None,
    eval_profile_fn: Any | None = None,
    promote_checkpoint_fn: Any | None = None,
) -> OperatorJobRecord:
    existing = _get_job(queue, job_id)
    if existing.status == "completed":
        return existing

    queue.update_job(job_id, status="running")
    active = _get_job(queue, job_id)
    payload = _validate_operator_job_payload(active.job_type, active.payload)

    try:
        if active.job_type == "train-profile":
            result = _run_train_profile_job(
                job_id=job_id,
                payload=payload,
                settings=settings,
                profile_registry=profile_registry,
                lifecycle_manager=lifecycle_manager,
                train_profile_fn=train_profile_fn,
            )
        elif active.job_type == "eval-profile":
            result = _run_eval_profile_job(
                payload=payload,
                settings=settings,
                lifecycle_manager=lifecycle_manager,
                eval_profile_fn=eval_profile_fn,
            )
        elif active.job_type == "promote-checkpoint":
            result = _run_promote_checkpoint_job(
                payload=payload,
                settings=settings,
                lifecycle_manager=lifecycle_manager,
                promote_checkpoint_fn=promote_checkpoint_fn,
            )
        else:
            raise ValueError(f"unsupported job_type: {active.job_type}")
    except Exception as exc:
        queue.update_job(job_id, status="failed", last_error=str(exc))
        raise

    normalized_result = _normalize_result(result)
    return queue.update_job(job_id, status="completed", result=normalized_result)


def run_continuous_training_cycle(
    queue: OperatorJobQueue,
    *,
    experiment_id: str,
    advisor_profile_id: str,
    rollout_group: dict,
    benchmark_manifests: list[dict],
    promotion_threshold: float = 0.05,
    settings: AdvisorSettings | None = None,
    profile_registry: AdvisorProfileRegistry | None = None,
    lifecycle_manager: CheckpointLifecycleManager | None = None,
    train_profile_fn: Any | None = None,
    eval_profile_fn: Any | None = None,
    promote_checkpoint_fn: Any | None = None,
) -> dict:
    cycle_key = f"continuous:{experiment_id}:{advisor_profile_id}"
    train_job = _enqueue_or_reuse_job(
        queue,
        job_type="train-profile",
        payload=TrainProfileJobPayload(
            experiment_id=experiment_id,
            advisor_profile_id=advisor_profile_id,
            rollout_group=rollout_group,
            benchmark_manifests=benchmark_manifests,
        ).model_dump(),
        resume_token=f"{cycle_key}:train",
    )
    train_record = run_operator_job(
        queue,
        train_job.job_id,
        settings=settings,
        profile_registry=profile_registry,
        lifecycle_manager=lifecycle_manager,
        train_profile_fn=train_profile_fn,
        eval_profile_fn=eval_profile_fn,
        promote_checkpoint_fn=promote_checkpoint_fn,
    )
    checkpoint_id = train_record.result.get("checkpoint_id")
    if not checkpoint_id:
        raise ValueError("train-profile cycle result must include checkpoint_id")

    eval_job = _enqueue_or_reuse_job(
        queue,
        job_type="eval-profile",
        payload=EvalProfileJobPayload(
            advisor_profile_id=advisor_profile_id,
            candidate_checkpoint_id=checkpoint_id,
            benchmark_manifests=benchmark_manifests,
            promotion_threshold=promotion_threshold,
        ).model_dump(),
        resume_token=f"{cycle_key}:eval:{checkpoint_id}",
    )
    eval_record = run_operator_job(
        queue,
        eval_job.job_id,
        settings=settings,
        profile_registry=profile_registry,
        lifecycle_manager=lifecycle_manager,
        train_profile_fn=train_profile_fn,
        eval_profile_fn=eval_profile_fn,
        promote_checkpoint_fn=promote_checkpoint_fn,
    )

    promote_record = None
    promoted = False
    if eval_record.result.get("promote") is True:
        promote_job = _enqueue_or_reuse_job(
            queue,
            job_type="promote-checkpoint",
            payload=PromoteCheckpointJobPayload(
                advisor_profile_id=advisor_profile_id,
                candidate_checkpoint_id=checkpoint_id,
                evaluation=eval_record.result,
            ).model_dump(),
            resume_token=f"{cycle_key}:promote:{checkpoint_id}",
        )
        promote_record = run_operator_job(
            queue,
            promote_job.job_id,
            settings=settings,
            profile_registry=profile_registry,
            lifecycle_manager=lifecycle_manager,
            train_profile_fn=train_profile_fn,
            eval_profile_fn=eval_profile_fn,
            promote_checkpoint_fn=promote_checkpoint_fn,
        )
        promoted = bool(promote_record.result.get("promoted"))

    return ContinuousTrainingCycleResult(
        train_job=train_record.model_dump(),
        eval_job=eval_record.model_dump(),
        promote_job=promote_record.model_dump() if promote_record is not None else None,
        promoted=promoted,
    ).model_dump()


def _run_train_profile_job(
    *,
    job_id: str,
    payload: TrainProfileJobPayload,
    settings: AdvisorSettings | None,
    profile_registry: AdvisorProfileRegistry | None,
    lifecycle_manager: CheckpointLifecycleManager | None,
    train_profile_fn: Any | None,
) -> dict:
    if train_profile_fn is not None:
        return _normalize_result(train_profile_fn(payload))

    if settings is None:
        raise ValueError("settings are required to execute train-profile jobs")
    registry = profile_registry or AdvisorProfileRegistry.from_toml(settings.advisor_profiles_path)
    manager = lifecycle_manager or _build_lifecycle_manager(settings)
    result = run_profile_training_job(
        job_id=job_id,
        experiment_id=payload.experiment_id,
        advisor_profile_id=payload.advisor_profile_id,
        rollout_group=TrainingRolloutGroupResult.model_validate(payload.rollout_group),
        profile_registry=registry,
        lifecycle_manager=manager,
    )
    return result.model_dump()


def _run_eval_profile_job(
    *,
    payload: EvalProfileJobPayload,
    settings: AdvisorSettings | None,
    lifecycle_manager: CheckpointLifecycleManager | None,
    eval_profile_fn: Any | None,
) -> dict:
    if eval_profile_fn is not None:
        return _normalize_result(eval_profile_fn(payload))

    if settings is None:
        raise ValueError("settings are required to execute eval-profile jobs")
    manager = lifecycle_manager or _build_lifecycle_manager(settings)
    result = evaluate_profile_checkpoint_for_promotion(
        advisor_profile_id=payload.advisor_profile_id,
        candidate_checkpoint_id=payload.candidate_checkpoint_id,
        benchmark_manifests=[BenchmarkRunManifest.model_validate(item) for item in payload.benchmark_manifests],
        lifecycle_manager=manager,
        promotion_threshold=payload.promotion_threshold,
    )
    return result.model_dump()


def _run_promote_checkpoint_job(
    *,
    payload: PromoteCheckpointJobPayload,
    settings: AdvisorSettings | None,
    lifecycle_manager: CheckpointLifecycleManager | None,
    promote_checkpoint_fn: Any | None,
) -> dict:
    evaluation = payload.evaluation or {}
    if evaluation.get("advisor_profile_id") != payload.advisor_profile_id:
        raise ValueError("promotion evaluation advisor_profile_id does not match payload")
    if evaluation.get("candidate_checkpoint_id") != payload.candidate_checkpoint_id:
        raise ValueError("promotion evaluation candidate_checkpoint_id does not match payload")
    if evaluation.get("promote") is not True:
        raise ValueError("promote-checkpoint requires prior passing evaluation evidence")

    if promote_checkpoint_fn is not None:
        return _normalize_result(promote_checkpoint_fn(payload))

    if settings is None:
        raise ValueError("settings are required to execute promote-checkpoint jobs")
    manager = lifecycle_manager or _build_lifecycle_manager(settings)
    existing = manager.get_checkpoint(payload.candidate_checkpoint_id)
    if existing is None:
        raise ValueError(f"unknown checkpoint_id: {payload.candidate_checkpoint_id}")
    if existing.status == "active":
        return {
            "promoted": False,
            "status": "noop",
            "checkpoint_id": existing.checkpoint_id,
            "advisor_profile_id": payload.advisor_profile_id,
        }
    promoted = manager.promote_checkpoint(payload.candidate_checkpoint_id)
    return {
        "promoted": True,
        "status": promoted.status,
        "checkpoint_id": promoted.checkpoint_id,
        "advisor_profile_id": promoted.advisor_profile_id,
    }


def _validate_operator_job_payload(job_type: str, payload: dict | BaseModel) -> BaseModel:
    if job_type not in SUPPORTED_OPERATOR_JOB_TYPES:
        raise ValueError(f"unsupported job_type: {job_type}")
    raw_payload = payload.model_dump() if isinstance(payload, BaseModel) else payload
    model_map = {
        "train-profile": TrainProfileJobPayload,
        "eval-profile": EvalProfileJobPayload,
        "promote-checkpoint": PromoteCheckpointJobPayload,
    }
    return model_map[job_type].model_validate(raw_payload)


def _normalize_result(result: Any) -> dict:
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise ValueError(f"operator job results must be dict-like, got {type(result).__name__}")


def _build_lifecycle_manager(settings: AdvisorSettings) -> CheckpointLifecycleManager:
    return CheckpointLifecycleManager(Path(settings.trace_db_path).expanduser().parent / "artifacts")


def _enqueue_or_reuse_job(
    queue: OperatorJobQueue,
    *,
    job_type: str,
    payload: dict,
    resume_token: str,
) -> OperatorJobRecord:
    for item in queue.list_jobs():
        if item.job_type == job_type and item.resume_token == resume_token:
            return item
    return queue.enqueue_job(job_type=job_type, payload=payload, resume_token=resume_token)


def _get_job(queue: OperatorJobQueue, job_id: str) -> OperatorJobRecord:
    for item in queue.list_jobs():
        if item.job_id == job_id:
            return item
    raise ValueError(f"unknown job_id: {job_id}")


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
