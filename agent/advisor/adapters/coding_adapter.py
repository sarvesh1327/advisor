from __future__ import annotations

from pathlib import Path

from agent.advisor.core.schemas import (
    AdvisorArtifact,
    AdvisorCapabilityDescriptor,
    AdvisorContext,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorTask,
    CandidateFile,
    FailureSignal,
    RepoSummary,
)

_EXCLUDED_PARTS = {"build", "dist", ".next", "coverage", "__pycache__"}


class CodingContextAdapter:
    def __init__(self, *, token_budget: int):
        self.token_budget = token_budget

    # This adapter maps repo/file/failure inputs into the generic packet while keeping legacy fields filled.
    def build_packet(
        self,
        *,
        run_id: str,
        task_text: str,
        task_type: str,
        repo: dict,
        file_tree_slice: list[str],
        candidate_files: list[CandidateFile],
        recent_failures: list[FailureSignal],
        constraints: list[str],
        tool_limits: dict,
        acceptance_criteria: list[str],
        changed_files: list[str],
    ) -> AdvisorInputPacket:
        prioritized_candidates = self._prioritize_candidates(candidate_files, changed_files)
        artifacts = [
            AdvisorArtifact(
                kind="file",
                locator=item.path,
                description=item.reason,
                metadata={
                    "source": "candidate_files",
                    "changed": item.path in changed_files,
                    "symbol_hint": self._symbol_hint(item.path),
                },
                score=item.score,
            )
            for item in prioritized_candidates
        ]
        history = [
            AdvisorHistoryEntry(
                kind=item.kind,
                summary=item.summary,
                locator=item.file,
                metadata={"fix_hint": item.fix_hint} if item.fix_hint else {},
            )
            for item in recent_failures
        ]
        context = AdvisorContext(
            summary="coding task context",
            metadata={
                "repo": repo,
                "repo_summary": {
                    "modules": self._modules_from_tree(file_tree_slice),
                    "hotspots": [item.path for item in prioritized_candidates[:3]],
                    "file_tree_slice": file_tree_slice,
                },
                "tool_limits": tool_limits,
                "token_budget": self.token_budget,
                "changed_files": changed_files,
            },
        )
        return AdvisorInputPacket(
            run_id=run_id,
            task_text=task_text,
            task_type=task_type,
            repo=repo,
            repo_summary=RepoSummary(
                modules=self._modules_from_tree(file_tree_slice),
                hotspots=[item.path for item in prioritized_candidates[:3]],
                file_tree_slice=file_tree_slice,
            ),
            candidate_files=prioritized_candidates,
            recent_failures=recent_failures,
            constraints=constraints,
            tool_limits=tool_limits,
            acceptance_criteria=acceptance_criteria,
            token_budget=self.token_budget,
            task=AdvisorTask(domain="coding", text=task_text, type=task_type),
            context=context,
            artifacts=artifacts,
            history=history,
            domain_capabilities=[
                AdvisorCapabilityDescriptor(
                    domain="coding",
                    supported_artifact_kinds=["file"],
                    supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
                    supports_changed_artifacts=True,
                    supports_symbol_regions=True,
                )
            ],
        )

    # Changed source files stay ahead of generated outputs.
    def _prioritize_candidates(self, candidate_files: list[CandidateFile], changed_files: list[str]) -> list[CandidateFile]:
        changed = set(changed_files)
        prioritized = []
        for item in candidate_files:
            if self._is_excluded(item.path):
                continue
            score = item.score + (0.25 if item.path in changed else 0.0)
            if self._is_source_file(item.path):
                score += 0.2
            prioritized.append(CandidateFile(path=item.path, reason=item.reason, score=score))
        prioritized.sort(key=lambda item: (-item.score, item.path))
        return prioritized

    def _is_excluded(self, path: str) -> bool:
        return bool(_EXCLUDED_PARTS & {part.lower() for part in Path(path).parts})

    def _is_source_file(self, path: str) -> bool:
        return Path(path).suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java"}

    def _symbol_hint(self, path: str) -> str | None:
        if not self._is_source_file(path):
            return None
        return Path(path).stem or None

    # Top-level path segments are the current lightweight module proxy for coding repos.
    def _modules_from_tree(self, file_tree: list[str]) -> list[str]:
        modules = []
        seen = set()
        for rel in file_tree:
            top = rel.split("/", 1)[0]
            if top not in seen:
                seen.add(top)
                modules.append(top)
        return modules[:10]
