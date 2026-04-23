from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, FailureSignal, RewardLabel


class AdvisorTraceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # Keep schema creation idempotent so local startup never depends on migration tooling.
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  started_at TEXT NOT NULL,
                  task_text TEXT NOT NULL,
                  task_type TEXT NOT NULL,
                  repo_path TEXT NOT NULL,
                  branch TEXT,
                  session_id TEXT,
                  task_id TEXT,
                  advisor_profile_id TEXT
                );
                CREATE TABLE IF NOT EXISTS run_contexts (
                  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                  repo_summary_json TEXT NOT NULL,
                  candidate_files_json TEXT NOT NULL,
                  recent_failures_json TEXT NOT NULL,
                  constraints_json TEXT NOT NULL,
                  tool_limits_json TEXT NOT NULL,
                  acceptance_criteria_json TEXT NOT NULL,
                  token_budget INTEGER NOT NULL,
                  task_json TEXT,
                  context_json TEXT,
                  artifacts_json TEXT,
                  history_json TEXT,
                  domain_capabilities_json TEXT
                );
                CREATE TABLE IF NOT EXISTS advice_records (
                  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                  advisor_model TEXT NOT NULL,
                  prompt_hash TEXT NOT NULL,
                  advice_json TEXT NOT NULL,
                  injected_advice_json TEXT,
                  injected_rendered_advice TEXT,
                  injection_policy_json TEXT,
                  latency_ms INTEGER NOT NULL,
                  validated INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_outcomes (
                  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                  status TEXT NOT NULL,
                  files_touched_json TEXT NOT NULL,
                  retries INTEGER NOT NULL,
                  tests_run_json TEXT NOT NULL,
                  review_verdict TEXT,
                  summary TEXT,
                  completed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS failure_patterns (
                  pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signature TEXT NOT NULL,
                  frequency INTEGER NOT NULL DEFAULT 1,
                  fix_hint TEXT,
                  last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_examples (
                  example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT REFERENCES runs(run_id),
                  input_json TEXT NOT NULL,
                  target_advice_json TEXT NOT NULL,
                  split TEXT NOT NULL,
                  quality_score REAL NOT NULL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS reward_labels (
                  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                  reward_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_lineages (
                  run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                  manifest_json TEXT NOT NULL,
                  lineage_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(run_contexts)").fetchall()
            }
            run_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "advisor_profile_id" not in run_columns:
                conn.execute("ALTER TABLE runs ADD COLUMN advisor_profile_id TEXT")
            for column_name in (
                "task_json",
                "context_json",
                "artifacts_json",
                "history_json",
                "domain_capabilities_json",
            ):
                if column_name not in columns:
                    conn.execute(f"ALTER TABLE run_contexts ADD COLUMN {column_name} TEXT")
            advice_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(advice_records)").fetchall()
            }
            for column_name in (
                "injected_advice_json",
                "injected_rendered_advice",
                "injection_policy_json",
            ):
                if column_name not in advice_columns:
                    conn.execute(f"ALTER TABLE advice_records ADD COLUMN {column_name} TEXT")

    def record_task_run(
        self,
        packet: AdvisorInputPacket,
        advice: AdviceBlock,
        *,
        advisor_model: str,
        advisor_profile_id: str = "default",
        latency_ms: int,
        prompt_hash: str,
        validated: bool = True,
        injected_advice: AdviceBlock | dict | None = None,
        injected_rendered_advice: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(run_id, started_at, task_text, task_type, repo_path, branch, session_id, task_id, advisor_profile_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.run_id,
                    now,
                    packet.task_text,
                    packet.task_type,
                    packet.repo.get("path", ""),
                    packet.repo.get("branch"),
                    packet.repo.get("session_id"),
                    packet.repo.get("task_id"),
                    advisor_profile_id,
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO run_contexts(
                    run_id, repo_summary_json, candidate_files_json, recent_failures_json,
                    constraints_json, tool_limits_json, acceptance_criteria_json, token_budget,
                    task_json, context_json, artifacts_json, history_json, domain_capabilities_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.run_id,
                    json.dumps(packet.repo_summary.model_dump()),
                    json.dumps([item.model_dump() for item in packet.candidate_files]),
                    json.dumps([item.model_dump() for item in packet.recent_failures]),
                    json.dumps(packet.constraints),
                    json.dumps(packet.tool_limits),
                    json.dumps(packet.acceptance_criteria),
                    packet.token_budget,
                    json.dumps(packet.task.model_dump() if packet.task else None),
                    json.dumps(packet.context.model_dump() if packet.context else None),
                    json.dumps([item.model_dump() for item in packet.artifacts]),
                    json.dumps([item.model_dump() for item in packet.history]),
                    json.dumps([item.model_dump() for item in packet.domain_capabilities]),
                ),
            )
            # Advice is stored after validation so downstream exports only see normalized output.
            conn.execute(
                """
                INSERT OR REPLACE INTO advice_records(
                    run_id, advisor_model, prompt_hash, advice_json, injected_advice_json,
                    injected_rendered_advice, injection_policy_json, latency_ms, validated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    packet.run_id,
                    advisor_model,
                    prompt_hash,
                    advice.model_dump_json(),
                    json.dumps(
                        injected_advice.model_dump()
                        if isinstance(injected_advice, AdviceBlock)
                        else injected_advice or advice.model_dump()
                    ),
                    injected_rendered_advice,
                    advice.injection_policy.model_dump_json(),
                    latency_ms,
                    1 if validated else 0,
                ),
            )

    def record_outcome(self, outcome: AdvisorOutcome) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_outcomes(
                    run_id, status, files_touched_json, retries, tests_run_json,
                    review_verdict, summary, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.run_id,
                    outcome.status,
                    json.dumps(outcome.files_touched),
                    outcome.retries,
                    json.dumps(outcome.tests_run),
                    outcome.review_verdict,
                    outcome.summary,
                    now,
                ),
            )
            if outcome.status != "success":
                # Failure signatures feed the lightweight failure-memory path in context building.
                signature = outcome.summary or "unknown failure"
                existing = conn.execute(
                    "SELECT pattern_id, frequency FROM failure_patterns WHERE signature = ?",
                    (signature,),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE failure_patterns SET frequency = ?, last_seen_at = ? WHERE pattern_id = ?",
                        (existing["frequency"] + 1, now, existing["pattern_id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO failure_patterns(signature, frequency, fix_hint, last_seen_at) VALUES (?, 1, NULL, ?)",
                        (signature, now),
                    )

    def record_reward_label(self, reward_label: RewardLabel | dict) -> None:
        now = datetime.now(UTC).isoformat()
        normalized = reward_label if isinstance(reward_label, RewardLabel) else RewardLabel.model_validate(reward_label)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reward_labels(run_id, reward_json, created_at)
                VALUES (?, ?, ?)
                """,
                (normalized.run_id, normalized.model_dump_json(), now),
            )

    def record_lineage(self, run_id: str, manifest, lineage) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_lineages(run_id, manifest_json, lineage_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, manifest.model_dump_json(), lineage.model_dump_json(), now),
            )

    def get_lineage(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT manifest_json, lineage_json FROM run_lineages WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "manifest": json.loads(row["manifest_json"]),
            "lineage": json.loads(row["lineage_json"]),
        }

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT r.*, rc.repo_summary_json, rc.candidate_files_json, rc.recent_failures_json,
                       rc.constraints_json, rc.tool_limits_json, rc.acceptance_criteria_json, rc.token_budget,
                       rc.task_json, rc.context_json, rc.artifacts_json, rc.history_json, rc.domain_capabilities_json,
                       ar.advice_json, ar.injected_advice_json, ar.injected_rendered_advice,
                       ar.injection_policy_json, ar.advisor_model, ar.latency_ms,
                       ro.status, ro.files_touched_json, ro.retries, ro.tests_run_json, ro.review_verdict, ro.summary,
                       rl.reward_json
                FROM runs r
                LEFT JOIN run_contexts rc ON rc.run_id = r.run_id
                LEFT JOIN advice_records ar ON ar.run_id = r.run_id
                LEFT JOIN run_outcomes ro ON ro.run_id = r.run_id
                LEFT JOIN reward_labels rl ON rl.run_id = r.run_id
                WHERE r.run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_run_dict(row, include_context=True)

    def list_runs(self, include_context: bool = False) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, rc.repo_summary_json, rc.candidate_files_json, rc.recent_failures_json,
                       rc.constraints_json, rc.tool_limits_json, rc.acceptance_criteria_json, rc.token_budget,
                       rc.task_json, rc.context_json, rc.artifacts_json, rc.history_json, rc.domain_capabilities_json,
                       ar.advice_json, ar.injected_advice_json, ar.injected_rendered_advice,
                       ar.injection_policy_json, ar.advisor_model, ar.latency_ms,
                       ro.status, ro.files_touched_json, ro.retries, ro.tests_run_json, ro.review_verdict, ro.summary,
                       rl.reward_json
                FROM runs r
                LEFT JOIN run_contexts rc ON rc.run_id = r.run_id
                LEFT JOIN advice_records ar ON ar.run_id = r.run_id
                LEFT JOIN run_outcomes ro ON ro.run_id = r.run_id
                LEFT JOIN reward_labels rl ON rl.run_id = r.run_id
                ORDER BY r.started_at DESC
                """
            ).fetchall()
        return [self._row_to_run_dict(row, include_context=include_context) for row in rows]

    def delete_runs(self, run_ids: list[str]) -> int:
        if not run_ids:
            return 0
        placeholders = ", ".join("?" for _ in run_ids)
        with self._connect() as conn:
            for table_name in (
                "training_examples",
                "run_lineages",
                "reward_labels",
                "run_outcomes",
                "advice_records",
                "run_contexts",
                "runs",
            ):
                conn.execute(f"DELETE FROM {table_name} WHERE run_id IN ({placeholders})", run_ids)
        return len(run_ids)

    def _row_to_run_dict(self, row: sqlite3.Row, *, include_context: bool) -> dict:
        result = {
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "task_text": row["task_text"],
            "task_type": row["task_type"],
            "repo_path": row["repo_path"],
            "branch": row["branch"],
            "session_id": row["session_id"],
            "task_id": row["task_id"],
            "advisor_profile_id": row["advisor_profile_id"],
            "advice": json.loads(row["advice_json"]) if row["advice_json"] else None,
            "injected_advice": json.loads(row["injected_advice_json"]) if row["injected_advice_json"] else None,
            "injected_rendered_advice": row["injected_rendered_advice"],
            "injection_policy": json.loads(row["injection_policy_json"]) if row["injection_policy_json"] else None,
            "reward_label": json.loads(row["reward_json"]) if row["reward_json"] else None,
            "outcome": {
                "status": row["status"],
                "files_touched": json.loads(row["files_touched_json"]) if row["files_touched_json"] else [],
                "retries": row["retries"] or 0,
                "tests_run": json.loads(row["tests_run_json"]) if row["tests_run_json"] else [],
                "review_verdict": row["review_verdict"],
                "summary": row["summary"],
            } if row["status"] else None,
        }
        if include_context:
            task_json = json.loads(row["task_json"]) if row["task_json"] else None
            context_json = json.loads(row["context_json"]) if row["context_json"] else None
            artifacts_json = json.loads(row["artifacts_json"]) if row["artifacts_json"] else []
            history_json = json.loads(row["history_json"]) if row["history_json"] else []
            capabilities_json = json.loads(row["domain_capabilities_json"]) if row["domain_capabilities_json"] else []
            # Rehydrate through the schema so replayed runs prefer stored generic state.
            packet = AdvisorInputPacket(
                run_id=row["run_id"],
                task_text=row["task_text"],
                task_type=row["task_type"],
                repo={"path": row["repo_path"], "branch": row["branch"], "dirty": False},
                repo_summary=json.loads(row["repo_summary_json"]) if row["repo_summary_json"] else {},
                candidate_files=json.loads(row["candidate_files_json"]) if row["candidate_files_json"] else [],
                recent_failures=json.loads(row["recent_failures_json"]) if row["recent_failures_json"] else [],
                constraints=json.loads(row["constraints_json"]) if row["constraints_json"] else [],
                tool_limits=json.loads(row["tool_limits_json"]) if row["tool_limits_json"] else {},
                acceptance_criteria=json.loads(row["acceptance_criteria_json"]) if row["acceptance_criteria_json"] else [],
                token_budget=row["token_budget"] or 0,
                task=task_json,
                context=context_json,
                artifacts=artifacts_json,
                history=history_json,
                domain_capabilities=capabilities_json,
            )
            result["input"] = packet.model_dump()
        return result

    def find_recent_failures(
        self,
        task_text: str,
        repo_path: str,
        limit: int = 5,
        changed_files: list[str] | None = None,
    ) -> list[FailureSignal]:
        task_tokens = {tok.lower() for tok in task_text.split() if len(tok) >= 4}
        changed = set(changed_files or [])
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.task_text, ro.summary, ro.review_verdict, ro.files_touched_json, ro.completed_at
                FROM run_outcomes ro
                JOIN runs r ON r.run_id = ro.run_id
                WHERE r.repo_path = ? AND ro.status != 'success'
                ORDER BY ro.completed_at DESC
                LIMIT 50
                """,
                (repo_path,),
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            row_tokens = {
                tok.lower()
                for tok in ((row["task_text"] or "") + " " + (row["summary"] or "")).split()
                if len(tok) >= 4
            }
            files_touched = set(json.loads(row["files_touched_json"]) if row["files_touched_json"] else [])
            score = float(len(task_tokens & row_tokens))
            if changed:
                score += 3.0 * len(changed & files_touched)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["completed_at"] or ""), reverse=False)
        return [
            FailureSignal(kind="recent-failure", summary=row["summary"] or "previous failure", fix_hint=row["review_verdict"])
            for _score, row in scored[:limit]
        ]
