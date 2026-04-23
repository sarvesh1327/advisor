from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.measurement import build_phase5_measurement_report
from agent.advisor.operators.operator_runtime import OperatorJobRecord
from agent.advisor.training.training_runtime import CheckpointLifecycleManager


class BenchmarkReleasePolicy(BaseModel):
    min_overall_lift: float = 0.0
    min_reward_label_coverage: float = 0.95
    min_lineage_coverage: float = 0.95
    max_open_divergences: int = 4


class DeploymentHardeningProfile(BaseModel):
    mode: Literal["single_tenant", "hosted"]
    auth: dict = Field(default_factory=dict)
    tenancy: dict = Field(default_factory=dict)
    isolation: dict = Field(default_factory=dict)
    operator_runbooks: list[str] = Field(default_factory=list)
    backup_paths: list[str] = Field(default_factory=list)
    alerting: dict = Field(default_factory=dict)


class Phase8ValidationPolicy(BaseModel):
    min_completed_cycles: int = 2
    min_promoted_cycles: int = 1
    min_best_overall_delta: float = 0.0
    require_active_checkpoint: bool = True
    require_rollback_coverage: bool = True
    max_failed_jobs: int = 0


def build_phase8_validation_report(
    *,
    lifecycle_manager: CheckpointLifecycleManager,
    job_records: list[OperatorJobRecord] | list[dict] | None = None,
    required_profiles: list[str] | None = None,
    policy: Phase8ValidationPolicy | None = None,
) -> dict:
    active_policy = policy or Phase8ValidationPolicy()
    measurement = build_phase5_measurement_report(
        lifecycle_manager=lifecycle_manager,
        job_records=job_records,
    )
    normalized_jobs = [item.model_dump() if isinstance(item, OperatorJobRecord) else dict(item) for item in (job_records or [])]
    resolved_required_profiles = list(required_profiles or sorted((measurement.get("profiles") or {}).keys()))
    available_profiles = measurement.get("profiles") or {}
    missing_profiles = [profile_id for profile_id in resolved_required_profiles if profile_id not in available_profiles]
    profile_reports = {}
    profile_failed_checks = []
    for profile_id in resolved_required_profiles:
        profile_report = _build_phase8_profile_report(
            profile_id,
            available_profiles.get(profile_id),
            active_policy,
        )
        profile_reports[profile_id] = profile_report
        profile_failed_checks.extend(f"{profile_id}:{name}" for name, item in profile_report["checks"].items() if not item["pass"])

    job_summary = _build_phase8_job_summary(normalized_jobs)
    failed_jobs_check = {
        "actual": job_summary["failed"],
        "threshold": active_policy.max_failed_jobs,
        "pass": job_summary["failed"] <= active_policy.max_failed_jobs,
    }
    failed_checks = []
    if missing_profiles:
        failed_checks.append("required_profiles")
    failed_checks.extend(profile_failed_checks)
    if not failed_jobs_check["pass"]:
        failed_checks.append("failed_jobs")
    return {
        "pass": not failed_checks,
        "failed_checks": failed_checks,
        "policy": active_policy.model_dump(),
        "required_profiles": resolved_required_profiles,
        "missing_profiles": missing_profiles,
        "job_summary": job_summary,
        "checks": {
            "failed_jobs": failed_jobs_check,
        },
        "profiles": profile_reports,
        "measurement": measurement,
    }


def evaluate_release_gate(report: dict, policy: BenchmarkReleasePolicy | None = None) -> dict:
    active_policy = policy or BenchmarkReleasePolicy()
    lift = float(
        ((report.get("canonical_study") or {}).get("lift_summary") or {}).get(
            "advisor_minus_baseline_overall_score", 0.0
        )
    )
    provenance = report.get("provenance_coverage") or {}
    reward_coverage = float(provenance.get("reward_label_coverage", 0.0))
    lineage_coverage = float(provenance.get("lineage_coverage", 0.0))
    open_divergences = sum(1 for item in report.get("paper_divergences") or [] if item.get("status") == "open")
    checks = {
        "overall_lift": _check_metric(lift, active_policy.min_overall_lift),
        "reward_label_coverage": _check_metric(reward_coverage, active_policy.min_reward_label_coverage),
        "lineage_coverage": _check_metric(lineage_coverage, active_policy.min_lineage_coverage),
        "open_divergences": {
            "actual": open_divergences,
            "threshold": active_policy.max_open_divergences,
            "pass": open_divergences <= active_policy.max_open_divergences,
        },
    }
    passed = all(item["pass"] for item in checks.values())
    return {
        "pass": passed,
        "checks": checks,
        "policy": active_policy.model_dump(),
        "failed_checks": [name for name, item in checks.items() if not item["pass"]],
    }


def build_alert_summary(release_verdict: dict) -> dict:
    failed_checks = release_verdict.get("failed_checks") or []
    if not failed_checks:
        return {
            "severity": "info",
            "summary": "Release gate passed.",
            "failed_checks": [],
        }
    severity = "critical" if len(failed_checks) >= 2 else "warning"
    return {
        "severity": severity,
        "summary": f"Release gate failed: {', '.join(failed_checks)}",
        "failed_checks": failed_checks,
    }


def build_deployment_hardening_profile(
    *,
    mode: Literal["single_tenant", "hosted"],
    state_root: str | Path,
) -> DeploymentHardeningProfile:
    root = Path(state_root).expanduser()
    backup_paths = [str(root / "advisor.db"), str(root / "events.jsonl"), str(root / "archive")]
    if mode == "hosted":
        return DeploymentHardeningProfile(
            mode=mode,
            auth={"mode": "external-auth-proxy", "requirements": ["authenticated identity", "signed tenant context"]},
            tenancy={"tenant_id_required": True, "operator_impersonation_requires_audit": True},
            isolation={"storage_strategy": "per-tenant-root", "session_boundary": "per-tenant process or scoped worker"},
            operator_runbooks=[
                "rotate credentials and verify auth proxy headers",
                "audit tenant-scoped storage roots before upgrades",
                "run benchmark release gate before deployment",
            ],
            backup_paths=backup_paths,
            alerting={"channels": ["stderr", "operator-webhook"], "critical_on_release_gate_failure": True},
        )
    return DeploymentHardeningProfile(
        mode=mode,
        auth={"mode": "local-boundary", "requirements": ["host access only"]},
        tenancy={"tenant_id_required": False, "operator_impersonation_requires_audit": False},
        isolation={"storage_strategy": "single-root", "session_boundary": "local process boundary"},
        operator_runbooks=[
            "backup local state before upgrades",
            "run regression gate before shipping new checkpoints",
            "verify archive rotation after retention jobs",
        ],
        backup_paths=backup_paths,
        alerting={"channels": ["stderr"], "critical_on_release_gate_failure": True},
    )


def lock_truth_surface_contract(path: str | Path) -> str:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "packet_schema": {"version": "v1", "source": "agent.advisor.core.schemas.AdvisorInputPacket"},
        "advice_schema": {"version": "v1", "source": "agent.advisor.core.schemas.AdviceBlock"},
        "benchmark_contract": {"version": "v1", "source": "agent.advisor.evaluation.benchmark.BenchmarkRunManifest"},
        "reward_contract": {"version": "v1", "source": "agent.advisor.core.schemas.RewardLabel"},
        "experiment_report_contract": {"version": "v1", "source": "agent.advisor.evaluation.results_pass.build_phase16_results_report"},
    }
    output_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def export_product_bundle(*, output_dir: str | Path, settings: AdvisorSettings) -> str:
    bundle_root = Path(output_dir).expanduser()
    state_root = bundle_root / "state"
    state_root.mkdir(parents=True, exist_ok=True)

    trace_db = Path(settings.trace_db_path).expanduser()
    event_log = Path(settings.event_log_path).expanduser()
    archive_dir = trace_db.parent / "archive"

    _copy_if_exists(trace_db, state_root / trace_db.name)
    _copy_if_exists(event_log, state_root / event_log.name)
    if archive_dir.exists():
        shutil.copytree(archive_dir, state_root / "archive", dirs_exist_ok=True)

    manifest = {
        "trace_db": trace_db.name,
        "event_log": event_log.name,
        "archive_dir": "archive",
    }
    (bundle_root / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lock_truth_surface_contract(bundle_root / "truth-surface-contract.json")
    return str(bundle_root)


def import_product_bundle(*, bundle_path: str | Path, target_root: str | Path) -> Path:
    bundle_root = Path(bundle_path).expanduser()
    destination_root = Path(target_root).expanduser()
    destination_root.mkdir(parents=True, exist_ok=True)
    state_src = bundle_root / "state"
    state_dst = destination_root / "state"
    if state_src.exists():
        shutil.copytree(state_src, state_dst, dirs_exist_ok=True)
    contract_src = bundle_root / "truth-surface-contract.json"
    if contract_src.exists():
        shutil.copy2(contract_src, destination_root / "truth-surface-contract.json")
    manifest_src = bundle_root / "bundle-manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, destination_root / "bundle-manifest.json")
    return destination_root


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _check_metric(actual: float, threshold: float) -> dict:
    return {
        "actual": round(actual, 4),
        "threshold": round(threshold, 4),
        "pass": actual >= threshold,
    }



def _build_phase8_job_summary(jobs: list[dict]) -> dict:
    counts = {
        "total": len(jobs),
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }
    for job in jobs:
        status = job.get("status")
        if status in counts:
            counts[status] += 1
    return counts



def _build_phase8_profile_report(profile_id: str, measurement_profile: dict | None, policy: Phase8ValidationPolicy) -> dict:
    if measurement_profile is None:
        return {
            "summary": {
                "checkpoint_count": 0,
                "cycle_count": 0,
                "promoted_cycle_count": 0,
                "active_checkpoint_id": None,
                "latest_checkpoint_id": None,
                "latest_overall_delta": None,
                "best_overall_delta": 0.0,
                "rollback_cycle_count": 0,
            },
            "checkpoint_history": [],
            "trend_history": [],
            "checks": {
                "completed_cycles": _check_count(0, policy.min_completed_cycles),
                "promoted_cycles": _check_count(0, policy.min_promoted_cycles),
                "best_overall_delta": _check_metric(0.0, policy.min_best_overall_delta),
                "active_checkpoint": _check_presence(None, required=policy.require_active_checkpoint),
                "rollback_coverage": _check_count(0, 1 if policy.require_rollback_coverage else 0),
            },
        }

    trend_history = list(measurement_profile.get("trend_history") or [])
    summary = dict(measurement_profile.get("summary") or {})
    checkpoint_history = list(measurement_profile.get("checkpoint_history") or [])
    rollback_checkpoint_ids = {
        item.get("checkpoint_id")
        for item in checkpoint_history
        if item.get("checkpoint_id") and (item.get("status") == "rolled_back" or item.get("rollback_reason"))
    }
    rollback_checkpoint_ids.update(
        item.get("checkpoint_id") for item in trend_history if item.get("checkpoint_id") and item.get("rollback") is True
    )
    rollback_cycle_count = len(rollback_checkpoint_ids)
    summary["rollback_cycle_count"] = rollback_cycle_count
    checks = {
        "completed_cycles": _check_count(int(summary.get("cycle_count") or 0), policy.min_completed_cycles),
        "promoted_cycles": _check_count(int(summary.get("promoted_cycle_count") or 0), policy.min_promoted_cycles),
        "best_overall_delta": _check_metric(float(summary.get("best_overall_delta") or 0.0), policy.min_best_overall_delta),
        "active_checkpoint": _check_presence(summary.get("active_checkpoint_id"), required=policy.require_active_checkpoint),
        "rollback_coverage": _check_count(rollback_cycle_count, 1 if policy.require_rollback_coverage else 0),
    }
    return {
        "summary": summary,
        "checkpoint_history": checkpoint_history,
        "trend_history": trend_history,
        "checks": checks,
    }



def _check_count(actual: int, minimum: int) -> dict:
    return {
        "actual": actual,
        "threshold": minimum,
        "pass": actual >= minimum,
    }



def _check_presence(value: str | None, *, required: bool) -> dict:
    if not required:
        return {
            "actual": value,
            "threshold": "optional",
            "pass": True,
        }
    return {
        "actual": value,
        "threshold": "present",
        "pass": bool(value),
    }
