from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.advisor.storage.trace_store import AdvisorTraceStore

_MAX_TITLE_CHARS = 90
_CONTEXT_BLOCK_RE = re.compile(r"<memory-context>.*?</memory-context>", re.IGNORECASE | re.DOTALL)
_ADVISOR_BLOCK_RE = re.compile(r"\[Advisor middleware\].*", re.IGNORECASE | re.DOTALL)
_SKILL_INVOKED_RE = re.compile(r"\[SYSTEM:\s*The user has invoked the [\"']([^\"']+)[\"'] skill", re.IGNORECASE)


def simplify_run_title(task_text: str | None, *, max_chars: int = _MAX_TITLE_CHARS) -> str:
    text = str(task_text or "").strip()
    if not text:
        return "Untitled task"
    skill_match = _SKILL_INVOKED_RE.search(text)
    if skill_match:
        return f"Skill: {skill_match.group(1)}"
    if text.startswith("[SYSTEM: Background process"):
        return "Background process update"
    if text.startswith("Review the conversation above and consider saving or updating a skill"):
        return "Skill maintenance check"
    text = _CONTEXT_BLOCK_RE.sub(" ", text)
    text = _ADVISOR_BLOCK_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0] if lines else "Untitled task"
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) <= max_chars:
        return title
    return title[: max_chars - 1].rstrip() + "…"


def build_advisor_activity_snapshot(
    store: AdvisorTraceStore,
    *,
    limit: int = 20,
    lifecycle_manager: Any | None = None,
    required_profiles: list[str] | None = None,
) -> dict:
    rows = store.list_runs(include_context=True)[:limit]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_runs": len(store.list_runs(include_context=False)),
        "evidence": build_advisor_evidence_snapshot(
            store,
            lifecycle_manager=lifecycle_manager,
            required_profiles=required_profiles,
        ),
        "runs": [_normalize_run(row, store=store) for row in rows],
    }


def build_advisor_evidence_snapshot(
    store: AdvisorTraceStore,
    *,
    lifecycle_manager: Any | None = None,
    required_profiles: list[str] | None = None,
) -> dict:
    rows = store.list_runs(include_context=False)
    run_ids = {row.get("run_id") for row in rows if row.get("run_id")}
    lineage_run_ids = store.list_lineage_run_ids(run_ids)
    trajectory_run_ids = {trajectory.get("run_id") for trajectory in store.list_trajectories() if trajectory.get("run_id") in run_ids}
    database_counts = {
        "runs": len(rows),
        "outcomes": sum(1 for row in rows if row.get("outcome")),
        "reward_labels": sum(1 for row in rows if row.get("reward_label")),
        "lineages": len(lineage_run_ids),
        "trajectories": len(trajectory_run_ids),
    }
    artifact_counts = {
        "checkpoints": 0,
        "active_checkpoints": 0,
        "adapter_files_present": 0,
        "training_manifests_present": 0,
        "backend_manifests_present": 0,
    }
    profile_reports: dict[str, dict] = {}
    if lifecycle_manager is not None:
        records = lifecycle_manager.list_checkpoints()
        artifact_counts = _artifact_counts(records)
        profile_ids = sorted(
            set(required_profiles or [])
            | {str(record.advisor_profile_id) for record in records if record.advisor_profile_id}
        )
        profile_reports = {
            profile_id: _profile_artifact_evidence(lifecycle_manager, profile_id)
            for profile_id in profile_ids
        }
    blocking_reasons = _evidence_blocking_reasons(database_counts, profile_reports)
    return {
        "blocked": bool(blocking_reasons),
        "blocking_reasons": blocking_reasons,
        "database_counts": database_counts,
        "artifact_counts": artifact_counts,
        "profiles": profile_reports,
    }


def render_advisor_activity_dashboard(snapshot: dict) -> str:
    cards = "\n".join(_render_run_card(run) for run in snapshot.get("runs", [])) or "<p>No Advisor runs recorded yet.</p>"
    generated_at = html.escape(str(snapshot.get("generated_at") or ""))
    total_runs = html.escape(str(snapshot.get("total_runs") or 0))
    evidence_panel = _render_evidence_panel(snapshot.get("evidence") or {})
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Advisor activity dashboard</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <meta http-equiv=\"refresh\" content=\"10\">
  <style>
    body {{ background: #0b1020; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .meta {{ color: #94a3b8; margin-bottom: 24px; }}
    .grid {{ display: grid; gap: 16px; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; }}
    .row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1f2937; color: #bfdbfe; margin-right: 8px; font-size: 12px; }}
    .pill-used {{ background: #064e3b; color: #bbf7d0; }}
    .pill-profile {{ background: #312e81; color: #ddd6fe; }}
    .pill-blocked {{ background: #7f1d1d; color: #fecaca; }}
    .pill-ok {{ background: #064e3b; color: #bbf7d0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; border-radius: 8px; padding: 12px; color: #e2e8f0; overflow-x: auto; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .muted {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Advisor activity dashboard</h1>
  <div class=\"meta\">Generated: {generated_at} · Total recorded runs: {total_runs}</div>
  {evidence_panel}
  <div class=\"grid\">{cards}</div>
</body>
</html>"""


def write_advisor_activity_dashboard(
    store: AdvisorTraceStore,
    output_path: str | Path,
    *,
    limit: int = 20,
    lifecycle_manager: Any | None = None,
    required_profiles: list[str] | None = None,
) -> dict:
    snapshot = build_advisor_activity_snapshot(
        store,
        limit=limit,
        lifecycle_manager=lifecycle_manager,
        required_profiles=required_profiles,
    )
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_advisor_activity_dashboard(snapshot), encoding="utf-8")
    return {
        "output_path": str(target),
        "generated_at": snapshot["generated_at"],
        "run_count": len(snapshot["runs"]),
        "total_runs": snapshot["total_runs"],
        "evidence": snapshot["evidence"],
    }


def _normalize_run(row: dict, *, store: AdvisorTraceStore) -> dict:
    advice = row.get("advice") or {}
    outcome = row.get("outcome") or {}
    reward = row.get("reward_label") or {}
    input_payload = row.get("input") or {}
    advisor_profile_id = row.get("advisor_profile_id") or "unknown"
    caller_id = row.get("caller_id") or "unknown"
    advisor_used = bool(advice or row.get("injected_advice") or row.get("injected_rendered_advice"))
    run_id = row.get("run_id")
    trajectory_present = bool(store.list_trajectories(run_id=run_id)) if run_id else False
    reward_present = bool(reward)
    lineage_present = store.get_lineage(run_id) is not None if run_id else False
    evidence_badges = {
        "trajectory": "yes" if trajectory_present else "no",
        "reward": "yes" if reward_present else "no",
        "lineage": "yes" if lineage_present else "no",
    }
    return {
        "run_id": run_id,
        "started_at": row.get("started_at"),
        "task_text": row.get("task_text"),
        "title": simplify_run_title(row.get("task_text")),
        "task_type": row.get("task_type"),
        "advisor_profile_id": advisor_profile_id,
        "caller_id": caller_id,
        "advisor_used": advisor_used,
        "profile_badge": f"Profile: {advisor_profile_id}",
        "caller_badge": f"Caller: {caller_id}",
        "repo_path": row.get("repo_path"),
        "recommended_plan": advice.get("recommended_plan") or [],
        "confidence": advice.get("confidence"),
        "injected_rendered_advice": row.get("injected_rendered_advice") or "",
        "outcome": outcome,
        "reward_label": {
            "total_reward": reward.get("total_reward"),
            "reward_profile_id": reward.get("reward_profile_id"),
            "example_type": reward.get("example_type"),
        }
        if reward
        else None,
        "evidence_badges": evidence_badges,
        "evidence_blocked": not all(value == "yes" for value in evidence_badges.values()),
        "request_packet": {
            "constraints": input_payload.get("constraints") or [],
            "acceptance_criteria": input_payload.get("acceptance_criteria") or [],
            "artifacts": input_payload.get("artifacts") or [],
        },
    }


def _render_evidence_panel(evidence: dict) -> str:
    if not evidence:
        return ""
    db_counts = evidence.get("database_counts") or {}
    artifact_counts = evidence.get("artifact_counts") or {}
    status = "blocked" if evidence.get("blocked") else "ready"
    status_class = "pill-blocked" if evidence.get("blocked") else "pill-ok"
    profile_items = "".join(
        f"<li>{html.escape(profile_id)}: active={html.escape(str(item.get('active_checkpoint_id') or 'none'))}, "
        f"adapter={html.escape('yes' if item.get('adapter_file_exists') else 'no')}, "
        f"training_manifest={html.escape('yes' if item.get('training_manifest_exists') else 'no')}, "
        f"backend_manifest={html.escape('yes' if item.get('backend_manifest_exists') else 'no')}</li>"
        for profile_id, item in (evidence.get("profiles") or {}).items()
    ) or "<li>No profile artifact evidence requested.</li>"
    return f"""
  <section class=\"card\">
    <h2>Evidence surface</h2>
    <span class=\"pill {status_class}\">evidence: {html.escape(status)}</span>
    <span class=\"pill\">runs: {html.escape(str(db_counts.get('runs', 0)))}</span>
    <span class=\"pill\">rewards: {html.escape(str(db_counts.get('reward_labels', 0)))}</span>
    <span class=\"pill\">lineages: {html.escape(str(db_counts.get('lineages', 0)))}</span>
    <span class=\"pill\">trajectories: {html.escape(str(db_counts.get('trajectories', 0)))}</span>
    <span class=\"pill\">active checkpoints: {html.escape(str(artifact_counts.get('active_checkpoints', 0)))}</span>
    <h3 style=\"margin-top:12px\">Profile artifacts</h3>
    <ul>{profile_items}</ul>
  </section>
    """


def _render_run_card(run: dict) -> str:
    plan_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("recommended_plan") or []) or "<li>No recommended plan recorded.</li>"
    constraints = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("request_packet", {}).get("constraints") or []) or "<li>None</li>"
    acceptance = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("request_packet", {}).get("acceptance_criteria") or []) or "<li>None</li>"
    outcome = run.get("outcome") or {}
    reward = run.get("reward_label") or {}
    advisor_used = "yes" if run.get("advisor_used") else "no"
    profile_badge = run.get("profile_badge") or f"Profile: {run.get('advisor_profile_id') or 'unknown'}"
    caller_badge = run.get("caller_badge") or f"Caller: {run.get('caller_id') or 'unknown'}"
    evidence_badges = run.get("evidence_badges") or {}
    evidence_blocked = "blocked" if run.get("evidence_blocked") else "ready"
    evidence_class = "pill-blocked" if run.get("evidence_blocked") else "pill-ok"
    return f"""
    <section class=\"card\">
      <h2>{html.escape(str(run.get('title') or simplify_run_title(run.get('task_text'))))}</h2>
      <div class=\"muted\">Run {html.escape(str(run.get('run_id') or ''))}</div>
      <div style=\"margin-top:8px\">
        <span class=\"pill pill-used\">Advisor used: {html.escape(advisor_used)}</span>
        <span class=\"pill pill-profile\">{html.escape(str(profile_badge))}</span>
        <span class=\"pill\">{html.escape(str(caller_badge))}</span>
        <span class=\"pill\">type: {html.escape(str(run.get('task_type') or ''))}</span>
        <span class=\"pill\">outcome: {html.escape(str(outcome.get('status') or 'pending'))}</span>
        <span class=\"pill\">reward: {html.escape(str(reward.get('total_reward') if reward else 'n/a'))}</span>
        <span class=\"pill\">trajectory: {html.escape(str(evidence_badges.get('trajectory') or 'n/a'))}</span>
        <span class=\"pill\">reward: {html.escape(str(evidence_badges.get('reward') or 'n/a'))}</span>
        <span class=\"pill\">lineage: {html.escape(str(evidence_badges.get('lineage') or 'n/a'))}</span>
        <span class=\"pill {evidence_class}\">evidence: {html.escape(evidence_blocked)}</span>
      </div>
      <div class=\"row\">
        <div>
          <h3>Request</h3>
          <div class=\"muted\">Repo: {html.escape(str(run.get('repo_path') or ''))}</div>
          <h3 style=\"margin-top:12px\">Constraints</h3>
          <ul>{constraints}</ul>
          <h3 style=\"margin-top:12px\">Acceptance criteria</h3>
          <ul>{acceptance}</ul>
        </div>
        <div>
          <h3>Advisor output</h3>
          <div class=\"muted\">Confidence: {html.escape(str(run.get('confidence') or 'n/a'))}</div>
          <ul>{plan_items}</ul>
          <h3 style=\"margin-top:12px\">Injected advice</h3>
          <pre>{html.escape(str(run.get('injected_rendered_advice') or ''))}</pre>
        </div>
      </div>
      <div class=\"row\">
        <div>
          <h3>Outcome summary</h3>
          <pre>{html.escape(str(outcome.get('summary') or 'No outcome summary yet.'))}</pre>
        </div>
        <div>
          <h3>Reward summary</h3>
          <pre>{html.escape(str(reward or {}))}</pre>
        </div>
      </div>
    </section>
    """


def _artifact_counts(records: list[Any]) -> dict:
    counts = {
        "checkpoints": len(records),
        "active_checkpoints": sum(1 for record in records if record.status == "active"),
        "adapter_files_present": 0,
        "training_manifests_present": 0,
        "backend_manifests_present": 0,
    }
    for record in records:
        artifact_paths = _checkpoint_artifact_paths(record)
        counts["adapter_files_present"] += int(_path_is_file(artifact_paths.get("adapter_model")))
        counts["training_manifests_present"] += int(_path_is_file(artifact_paths.get("training_manifest")))
        counts["backend_manifests_present"] += int(_path_is_file(artifact_paths.get("backend_manifest")))
    return counts


def _profile_artifact_evidence(lifecycle_manager: Any, profile_id: str) -> dict:
    records = lifecycle_manager.list_checkpoints(advisor_profile_id=profile_id)
    active_records = [record for record in records if record.status == "active"]
    selected = active_records[0] if active_records else (records[-1] if records else None)
    artifact_paths = _checkpoint_artifact_paths(selected) if selected is not None else {}
    active_checkpoint_id = active_records[0].checkpoint_id if active_records else None
    report = {
        "checkpoint_count": len(records),
        "active_checkpoint_id": active_checkpoint_id,
        "latest_checkpoint_id": selected.checkpoint_id if selected is not None else None,
        "adapter_model_path": artifact_paths.get("adapter_model"),
        "adapter_file_exists": _path_is_file(artifact_paths.get("adapter_model")),
        "adapter_config_path": artifact_paths.get("adapter_config"),
        "adapter_config_exists": _path_is_file(artifact_paths.get("adapter_config")),
        "training_manifest_path": artifact_paths.get("training_manifest"),
        "training_manifest_exists": _path_is_file(artifact_paths.get("training_manifest")),
        "backend_manifest_path": artifact_paths.get("backend_manifest"),
        "backend_manifest_exists": _path_is_file(artifact_paths.get("backend_manifest")),
    }
    blocking_reasons = []
    if report["active_checkpoint_id"] is None:
        blocking_reasons.append("missing_active_checkpoint")
    if not report["adapter_file_exists"]:
        blocking_reasons.append("missing_adapter_file")
    if not report["training_manifest_exists"]:
        blocking_reasons.append("missing_training_manifest")
    if not report["backend_manifest_exists"]:
        blocking_reasons.append("missing_backend_manifest")
    report["blocked"] = bool(blocking_reasons)
    report["blocking_reasons"] = blocking_reasons
    return report


def _checkpoint_artifact_paths(record: Any | None) -> dict:
    if record is None:
        return {}
    manifest_path = Path(record.path).expanduser() / "checkpoint.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(manifest.get("artifact_paths") or {})


def _path_is_file(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).expanduser().is_file()


def _evidence_blocking_reasons(database_counts: dict, profile_reports: dict[str, dict]) -> list[str]:
    reasons = []
    run_count = int(database_counts.get("runs") or 0)
    if run_count > int(database_counts.get("reward_labels") or 0):
        reasons.append("missing_reward_labels")
    if run_count > int(database_counts.get("lineages") or 0):
        reasons.append("missing_lineages")
    if run_count > int(database_counts.get("trajectories") or 0):
        reasons.append("missing_trajectories")
    for profile_id, report in profile_reports.items():
        if report.get("blocked"):
            reasons.append(f"profile_artifacts:{profile_id}")
    return reasons
