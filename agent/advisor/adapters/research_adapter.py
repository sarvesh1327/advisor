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


class ResearchContextAdapter:
    def __init__(self, *, token_budget: int):
        self.token_budget = token_budget

    # Research packets stay generic but label documents, notes, and sources explicitly for the runtime.
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
        prioritized_candidates = self._prioritize_candidates(task_text, candidate_files, changed_files)
        artifacts = [self._artifact_from_candidate(item, changed_files) for item in prioritized_candidates]
        history = [
            AdvisorHistoryEntry(
                kind=item.kind,
                summary=item.summary,
                locator=item.file,
                metadata={"fix_hint": item.fix_hint} if item.fix_hint else {},
            )
            for item in recent_failures
        ]
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
            task=AdvisorTask(domain="research-writing", text=task_text, type=task_type),
            context=AdvisorContext(
                summary="research-writing task context",
                metadata={
                    "repo": repo,
                    "changed_files": changed_files,
                    "artifact_counts": self._artifact_counts(artifacts),
                    "tool_limits": tool_limits,
                    "token_budget": self.token_budget,
                },
            ),
            artifacts=artifacts,
            history=history,
            domain_capabilities=[
                AdvisorCapabilityDescriptor(
                    domain="research-writing",
                    supported_artifact_kinds=["document", "note", "source"],
                    supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
                    supports_changed_artifacts=True,
                    supports_symbol_regions=False,
                )
            ],
        )

    # Prefer notes and sources when the task asks for grounding rather than draft polish.
    def _prioritize_candidates(self, task_text: str, candidate_files: list[CandidateFile], changed_files: list[str]) -> list[CandidateFile]:
        changed = set(changed_files)
        text = task_text.lower()
        prioritized = []
        for item in candidate_files:
            kind = self._artifact_kind(item.path)
            score = item.score + (0.3 if item.path in changed else 0.0)
            if "note" in text and kind == "note":
                score += 0.5
            if "source" in text and kind == "source":
                score += 0.45
            if any(token in text for token in ("draft", "memo", "brief")) and kind == "document":
                score += 0.1
            prioritized.append(CandidateFile(path=item.path, reason=item.reason, score=score))
        prioritized.sort(key=lambda item: (-item.score, item.path))
        return prioritized

    def _artifact_from_candidate(self, item: CandidateFile, changed_files: list[str]) -> AdvisorArtifact:
        kind = self._artifact_kind(item.path)
        return AdvisorArtifact(
            kind=kind,
            locator=item.path,
            description=item.reason,
            metadata={"source": "candidate_files", "changed": item.path in set(changed_files)},
            score=item.score,
        )

    def _artifact_kind(self, path: str) -> str:
        parts = Path(path).parts
        if "notes" in parts:
            return "note"
        if "sources" in parts or "references" in parts:
            return "source"
        return "document"

    def _artifact_counts(self, artifacts: list[AdvisorArtifact]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in artifacts:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def _modules_from_tree(self, file_tree: list[str]) -> list[str]:
        modules = []
        seen = set()
        for rel in file_tree:
            top = rel.split("/", 1)[0]
            if top not in seen:
                seen.add(top)
                modules.append(top)
        return modules[:10]
