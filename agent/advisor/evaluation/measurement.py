from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.advisor.operators.operator_runtime import OperatorJobRecord
from agent.advisor.training.training_runtime import CheckpointLifecycleManager, TrainingCheckpointRecord


# Phase 5 needs one profile-local report surface over checkpoint lineage and cycle history.
def build_phase5_measurement_report(
    *,
    lifecycle_manager: CheckpointLifecycleManager,
    job_records: list[OperatorJobRecord] | list[dict] | None = None,
) -> dict:
    checkpoint_records = [TrainingCheckpointRecord.model_validate(item) for item in lifecycle_manager._load_registry()]
    normalized_jobs = [item.model_dump() if isinstance(item, OperatorJobRecord) else dict(item) for item in (job_records or [])]
    profile_ids = sorted(
        {
            *(record.advisor_profile_id for record in checkpoint_records if record.advisor_profile_id),
            *(_job_profile_id(job) for job in normalized_jobs if _job_profile_id(job)),
        }
    )
    profiles = {
        profile_id: {
            "summary": _build_profile_summary(profile_id, checkpoint_records, normalized_jobs),
            "checkpoint_history": _build_checkpoint_history(profile_id, checkpoint_records),
            "trend_history": _build_trend_history(profile_id, normalized_jobs),
        }
        for profile_id in profile_ids
    }
    return {
        "profile_count": len(profiles),
        "profiles": profiles,
    }



def _build_profile_summary(profile_id: str, checkpoints: list[TrainingCheckpointRecord], jobs: list[dict]) -> dict:
    profile_checkpoints = [item for item in checkpoints if item.advisor_profile_id == profile_id]
    trend_history = _build_trend_history(profile_id, jobs)
    active = next((item for item in profile_checkpoints if item.status == "active"), None)
    latest_cycle = trend_history[-1] if trend_history else None
    return {
        "checkpoint_count": len(profile_checkpoints),
        "cycle_count": len(trend_history),
        "promoted_cycle_count": sum(1 for item in trend_history if item.get("promoted") is True),
        "active_checkpoint_id": active.checkpoint_id if active else None,
        "latest_checkpoint_id": latest_cycle.get("checkpoint_id") if latest_cycle else None,
        "latest_overall_delta": ((latest_cycle or {}).get("eval_delta") or {}).get("overall_score"),
        "best_overall_delta": max(
            (((item.get("eval_delta") or {}).get("overall_score") or 0.0) for item in trend_history),
            default=0.0,
        ),
    }



def _build_checkpoint_history(profile_id: str, checkpoints: list[TrainingCheckpointRecord]) -> list[dict]:
    profile_checkpoints = [item for item in checkpoints if item.advisor_profile_id == profile_id]
    history = []
    previous_fingerprint = None
    for item in profile_checkpoints:
        metadata = _load_checkpoint_manifest(item.path)
        fingerprint = _artifact_fingerprint(metadata)
        history.append(
            {
                "checkpoint_id": item.checkpoint_id,
                "experiment_id": item.experiment_id,
                "status": item.status,
                "rollback_reason": item.rollback_reason,
                "benchmark_summary": item.benchmark_summary,
                "artifact_fingerprint": fingerprint,
                "artifact_changed_vs_previous": None if previous_fingerprint is None else fingerprint != previous_fingerprint,
            }
        )
        previous_fingerprint = fingerprint
    return history



def _build_trend_history(profile_id: str, jobs: list[dict]) -> list[dict]:
    train_jobs = [
        job
        for job in jobs
        if job.get("job_type") == "train-profile"
        and job.get("status") == "completed"
        and ((job.get("payload") or {}).get("advisor_profile_id") == profile_id)
    ]
    ordered = sorted(train_jobs, key=lambda item: (item.get("created_at") or "", item.get("job_id") or ""))
    trend_rows = []
    for train_job in ordered:
        payload = train_job.get("payload") or {}
        result = train_job.get("result") or {}
        checkpoint_id = result.get("checkpoint_id")
        experiment_id = payload.get("experiment_id") or _cycle_experiment_id(train_job)
        eval_job = _find_completed_job(
            jobs,
            job_type="eval-profile",
            profile_id=profile_id,
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
        )
        promote_job = _find_completed_job(
            jobs,
            job_type="promote-checkpoint",
            profile_id=profile_id,
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
        )
        eval_result = (eval_job or {}).get("result") or {}
        promote_result = (promote_job or {}).get("result") or {}
        trend_rows.append(
            {
                "experiment_id": experiment_id,
                "checkpoint_id": checkpoint_id,
                "train_job_id": train_job.get("job_id"),
                "eval_job_id": (eval_job or {}).get("job_id"),
                "promote_job_id": (promote_job or {}).get("job_id"),
                "promotion_threshold": eval_result.get("promotion_threshold"),
                "eval_delta": dict(eval_result.get("deltas") or {}),
                "candidate_summary": dict(eval_result.get("candidate_summary") or {}),
                "baseline_summary": dict(eval_result.get("baseline_summary") or {}),
                "promote_decision": eval_result.get("promote"),
                "rollback": eval_result.get("rollback"),
                "decision_reason": eval_result.get("decision_reason"),
                "promoted": _resolve_promoted(eval_result, promote_result),
            }
        )
    return trend_rows



def _find_completed_job(
    jobs: list[dict],
    *,
    job_type: str,
    profile_id: str,
    checkpoint_id: str | None,
    experiment_id: str | None,
) -> dict | None:
    for job in sorted(jobs, key=lambda item: (item.get("created_at") or "", item.get("job_id") or "")):
        if job.get("job_type") != job_type or job.get("status") != "completed":
            continue
        payload = job.get("payload") or {}
        job_profile_id = _job_profile_id(job)
        if job_profile_id != profile_id:
            continue
        if checkpoint_id and _job_checkpoint_id(job) != checkpoint_id:
            continue
        job_experiment_id = _cycle_experiment_id(job)
        if experiment_id and job_experiment_id and job_experiment_id != experiment_id:
            continue
        if experiment_id and not job_experiment_id and payload.get("experiment_id") not in (None, experiment_id):
            continue
        return job
    return None



def _job_profile_id(job: dict) -> str | None:
    payload = job.get("payload") or {}
    result = job.get("result") or {}
    return payload.get("advisor_profile_id") or result.get("advisor_profile_id")



def _job_checkpoint_id(job: dict) -> str | None:
    payload = job.get("payload") or {}
    result = job.get("result") or {}
    return (
        result.get("checkpoint_id")
        or result.get("candidate_checkpoint_id")
        or payload.get("candidate_checkpoint_id")
        or payload.get("checkpoint_id")
    )



def _cycle_experiment_id(job: dict) -> str | None:
    payload = job.get("payload") or {}
    if payload.get("experiment_id"):
        return payload.get("experiment_id")
    resume_token = job.get("resume_token") or ""
    parts = resume_token.split(":")
    if len(parts) < 4 or parts[0] != "continuous":
        return None
    if parts[-1] == "train":
        return ":".join(parts[1:-2])
    if len(parts) >= 5 and parts[-2] in {"eval", "promote"}:
        return ":".join(parts[1:-3])
    return None



def _load_checkpoint_manifest(checkpoint_path: str) -> dict:
    manifest_path = Path(checkpoint_path).expanduser() / "checkpoint.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))



def _artifact_fingerprint(manifest: dict) -> str | None:
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    preferred = artifact_paths.get("adapter_model")
    if preferred:
        return _hash_path(preferred)
    for path in artifact_paths.values():
        fingerprint = _hash_path(path)
        if fingerprint is not None:
            return fingerprint
    return None



def _hash_path(path: str) -> str | None:
    target = Path(path).expanduser()
    if not target.exists():
        return None
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest[:16]



def _resolve_promoted(eval_result: dict, promote_result: dict) -> bool:
    if eval_result.get("promote") is not True:
        return False
    if not promote_result:
        return True
    if promote_result.get("status") == "noop":
        return True
    return bool(promote_result.get("promoted", True))
