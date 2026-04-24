from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Protocol

from agent.advisor.adapters.coding_adapter import CodingContextAdapter
from agent.advisor.adapters.conversation_adapter import ConversationContextAdapter
from agent.advisor.adapters.image_adapter import ImageUIContextAdapter, TextUIContextAdapter
from agent.advisor.adapters.research_adapter import ResearchContextAdapter
from agent.advisor.core.schemas import AdvisorInputPacket, CandidateFile
from agent.advisor.storage.trace_store import AdvisorTraceStore

_SKIP_DIRS = {".git", ".next", ".turbo", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}
_ENTRYPOINT_FILENAMES = {"main.py", "app.py", "server.py", "gateway.py", "cli.py", "api.py"}


class PacketAdapter(Protocol):
    def build_packet(self, **kwargs) -> AdvisorInputPacket:
        ...


class ContextBuilder:
    def __init__(
        self,
        trace_store: AdvisorTraceStore,
        max_tree_entries: int = 60,
        max_candidate_files: int = 8,
        max_failures: int = 5,
        token_budget: int = 1800,
        packet_adapter: PacketAdapter | None = None,
    ):
        self.trace_store = trace_store
        self.max_tree_entries = max_tree_entries
        self.max_candidate_files = max_candidate_files
        self.max_failures = max_failures
        self.token_budget = token_budget
        self.default_packet_adapter = packet_adapter
        self.packet_adapter = packet_adapter or CodingContextAdapter(token_budget=token_budget)
        self.adapter_registry = {
            "coding": CodingContextAdapter(token_budget=token_budget),
            "research-writing": ResearchContextAdapter(token_budget=token_budget),
            "text-ui": TextUIContextAdapter(token_budget=token_budget),
            "image-ui": ImageUIContextAdapter(token_budget=token_budget),
            "conversation": ConversationContextAdapter(token_budget=token_budget),
        }

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
        changed_files: list[str] | None = None,
        profile_domain: str | None = None,
    ) -> AdvisorInputPacket:
        repo = Path(repo_path).expanduser().resolve()
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        task_type = task_type_hint or self._infer_task_type(task_text)
        branch = branch or self._git_branch(repo)
        tree_slice = self._file_tree_slice(repo)
        candidates = self._candidate_files(task_text, tree_slice)
        failures = self.trace_store.find_recent_failures(
            task_text,
            str(repo),
            limit=self.max_failures,
            changed_files=changed_files or [],
        )
        constraints = self._constraints_from_task(task_text)
        adapter = self._select_adapter(
            task_text=task_text,
            task_type=task_type,
            tool_limits=tool_limits,
            profile_domain=profile_domain,
        )
        packet = adapter.build_packet(
            run_id=run_id,
            task_text=task_text,
            task_type=task_type,
            repo={"path": str(repo), "branch": branch, "dirty": self._is_dirty(repo)},
            file_tree_slice=tree_slice,
            candidate_files=candidates,
            recent_failures=failures,
            constraints=constraints,
            tool_limits=tool_limits,
            acceptance_criteria=acceptance_criteria or [],
            changed_files=changed_files or [],
        )
        if isinstance(packet, AdvisorInputPacket):
            return self._pack_packet(packet)
        return packet

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
        if any(tok in text for tok in ("image", "screenshot", "mockup", "visual")):
            return "ui-update"
        return "feature"

    def _select_adapter(
        self,
        *,
        task_text: str,
        task_type: str,
        tool_limits: dict,
        profile_domain: str | None = None,
    ) -> PacketAdapter:
        if self.default_packet_adapter is not None:
            return self.packet_adapter
        if profile_domain in self.adapter_registry:
            # Explicit profile selection should beat task-text heuristics.
            return self.adapter_registry[profile_domain]
        text = task_text.lower()
        if task_type in {"research", "analysis"} or any(tok in text for tok in ("research", "sources", "citations", "notes")):
            return self.adapter_registry["research-writing"]
        if task_type in {"ui-update", "image"} or tool_limits.get("image_read") or any(
            tok in text for tok in ("image", "screenshot", "mockup", "visual")
        ):
            return self.adapter_registry["image-ui"]
        return self.adapter_registry["coding"]

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

    def _pack_packet(self, packet: AdvisorInputPacket) -> AdvisorInputPacket:
        artifact_limit = self._artifact_limit()
        history_limit = self._history_limit()
        packet.candidate_files = packet.candidate_files[:artifact_limit]
        packet.artifacts = packet.artifacts[:artifact_limit]
        packet.history = packet.history[:history_limit]
        if packet.context is not None:
            packet.context.metadata["packed"] = {
                "candidate_files": len(packet.candidate_files),
                "artifacts": len(packet.artifacts),
                "history": len(packet.history),
            }
        return packet

    def _artifact_limit(self) -> int:
        if self.token_budget <= 360:
            return 1
        return max(1, min(self.max_candidate_files, self.token_budget // 220))

    def _history_limit(self) -> int:
        if self.token_budget <= 420:
            return 0
        return min(self.max_failures, max(1, (self.token_budget - 320) // 260))

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
