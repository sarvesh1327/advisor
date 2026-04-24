from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from pathlib import Path

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


def build_advisor_activity_snapshot(store: AdvisorTraceStore, *, limit: int = 20) -> dict:
    rows = store.list_runs(include_context=True)[:limit]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_runs": len(store.list_runs(include_context=False)),
        "runs": [_normalize_run(row) for row in rows],
    }


def render_advisor_activity_dashboard(snapshot: dict) -> str:
    cards = "\n".join(_render_run_card(run) for run in snapshot.get("runs", [])) or "<p>No Advisor runs recorded yet.</p>"
    generated_at = html.escape(str(snapshot.get("generated_at") or ""))
    total_runs = html.escape(str(snapshot.get("total_runs") or 0))
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
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; border-radius: 8px; padding: 12px; color: #e2e8f0; overflow-x: auto; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .muted {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Advisor activity dashboard</h1>
  <div class=\"meta\">Generated: {generated_at} · Total recorded runs: {total_runs}</div>
  <div class=\"grid\">{cards}</div>
</body>
</html>"""


def write_advisor_activity_dashboard(store: AdvisorTraceStore, output_path: str | Path, *, limit: int = 20) -> dict:
    snapshot = build_advisor_activity_snapshot(store, limit=limit)
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_advisor_activity_dashboard(snapshot), encoding="utf-8")
    return {
        "output_path": str(target),
        "generated_at": snapshot["generated_at"],
        "run_count": len(snapshot["runs"]),
        "total_runs": snapshot["total_runs"],
    }


def _normalize_run(row: dict) -> dict:
    advice = row.get("advice") or {}
    outcome = row.get("outcome") or {}
    reward = row.get("reward_label") or {}
    input_payload = row.get("input") or {}
    advisor_profile_id = row.get("advisor_profile_id") or "unknown"
    caller_id = row.get("caller_id") or "unknown"
    advisor_used = bool(advice or row.get("injected_advice") or row.get("injected_rendered_advice"))
    return {
        "run_id": row.get("run_id"),
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
        "request_packet": {
            "constraints": input_payload.get("constraints") or [],
            "acceptance_criteria": input_payload.get("acceptance_criteria") or [],
            "artifacts": input_payload.get("artifacts") or [],
        },
    }


def _render_run_card(run: dict) -> str:
    plan_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("recommended_plan") or []) or "<li>No recommended plan recorded.</li>"
    constraints = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("request_packet", {}).get("constraints") or []) or "<li>None</li>"
    acceptance = "".join(f"<li>{html.escape(str(item))}</li>" for item in run.get("request_packet", {}).get("acceptance_criteria") or []) or "<li>None</li>"
    outcome = run.get("outcome") or {}
    reward = run.get("reward_label") or {}
    advisor_used = "yes" if run.get("advisor_used") else "no"
    profile_badge = run.get("profile_badge") or f"Profile: {run.get('advisor_profile_id') or 'unknown'}"
    caller_badge = run.get("caller_badge") or f"Caller: {run.get('caller_id') or 'unknown'}"
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
