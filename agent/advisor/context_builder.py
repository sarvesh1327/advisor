from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path

from .schemas import AdvisorInputPacket, CandidateFile, RepoSummary
from .trace_store import AdvisorTraceStore

_SKIP_DIRS = {".git", ".next", ".turbo", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}
_ENTRYPOINT_FILENAMES = {"main.py", "app.py", "server.py", "gateway.py", "cli.py", "api.py"}


class ContextBuilder:
    def __init__(
        self,
        trace_store: AdvisorTraceStore,
        max_tree_entries: int = 60,
        max_candidate_files: int = 8,
        max_failures: int = 5,
        token_budget: int = 1800,
    ):
        self.trace_store = trace_store
        self.max_tree_entries = max_tree_entries
        self.max_candidate_files = max_candidate_files
        self.max_failures = max_failures
        self.token_budget = token_budget

    def build(
        self,
        *,
        task_text: str,
        repo_path: str,
        tool_limits: dict,
        acceptance_criteria: list[str] | None = None,
        run_id: str | None = None,
        branch: str | None = None,
        task_type_hint: str | None = None,
    ) -> AdvisorInputPacket:
        # This builder is the current coding adapter until packet construction is split by domain.
        repo = Path(repo_path).expanduser().resolve()
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        task_type = task_type_hint or self._infer_task_type(task_text)
        branch = branch or self._git_branch(repo)
        tree_slice = self._file_tree_slice(repo)
        candidates = self._candidate_files(task_text, tree_slice)
        failures = self.trace_store.find_recent_failures(task_text, str(repo), limit=self.max_failures)
        modules = self._modules_from_tree(tree_slice)
        constraints = self._constraints_from_task(task_text)
        # The coding adapter still fills the legacy fields; schemas.py backfills the generic packet view.
        return AdvisorInputPacket(
            run_id=run_id,
            task_text=task_text,
            task_type=task_type,
            repo={"path": str(repo), "branch": branch, "dirty": self._is_dirty(repo)},
            repo_summary=RepoSummary(modules=modules, hotspots=[c.path for c in candidates[:3]], file_tree_slice=tree_slice),
            candidate_files=candidates,
            recent_failures=failures,
            constraints=constraints,
            tool_limits=tool_limits,
            acceptance_criteria=acceptance_criteria or [],
            token_budget=self.token_budget,
        )

    def _infer_task_type(self, task_text: str) -> str:
        text = task_text.lower()
        if any(tok in text for tok in ("fix", "bug", "failing", "regression", "error")):
            return "bugfix"
        if any(tok in text for tok in ("refactor", "cleanup", "simplify")):
            return "refactor"
        if any(tok in text for tok in ("review", "audit")):
            return "review"
        if any(tok in text for tok in ("research", "investigate")):
            return "research"
        return "feature"

    def _file_tree_slice(self, repo: Path) -> list[str]:
        items: list[str] = []
        for root, dirs, files in os.walk(repo):
            # Prune generated directories early so they never pollute ranking.
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in sorted(files):
                if name.startswith('.'):
                    continue
                rel = str((Path(root) / name).relative_to(repo))
                items.append(rel)
                if len(items) >= self.max_tree_entries:
                    return items
        return items

    def _candidate_files(self, task_text: str, file_tree: list[str]) -> list[CandidateFile]:
        tokens = {tok for tok in re.findall(r"[a-zA-Z0-9_]+", task_text.lower()) if len(tok) >= 3}
        scored: list[tuple[float, str, str]] = []
        for rel in file_tree:
            rel_lower = rel.lower()
            matches = [tok for tok in tokens if tok in rel_lower]
            score = float(len(matches))
            if Path(rel).name in _ENTRYPOINT_FILENAMES:
                score += 0.5
            if score > 0:
                reason = f"matched task tokens: {', '.join(matches[:4])}" if matches else "likely entrypoint or hotspot"
                scored.append((score, rel, reason))
        if not scored:
            scored = [(0.1, rel, "default repo slice candidate") for rel in file_tree[: self.max_candidate_files]]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [CandidateFile(path=rel, reason=reason, score=score) for score, rel, reason in scored[: self.max_candidate_files]]

    def _modules_from_tree(self, file_tree: list[str]) -> list[str]:
        modules = []
        seen = set()
        for rel in file_tree:
            top = rel.split('/', 1)[0]
            if top not in seen:
                seen.add(top)
                modules.append(top)
        return modules[:10]

    def _constraints_from_task(self, task_text: str) -> list[str]:
        text = task_text.lower()
        constraints = []
        if "don't" in text or "do not" in text:
            constraints.append("respect explicit user prohibitions in task text")
        if "public api" in text:
            constraints.append("avoid changing public API unless necessary")
        return constraints

    def _git_branch(self, repo: Path) -> str | None:
        try:
            out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True)
            return out.strip() or None
        except Exception:
            return None

    def _is_dirty(self, repo: Path) -> bool:
        try:
            out = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True)
            return bool(out.strip())
        except Exception:
            return False
