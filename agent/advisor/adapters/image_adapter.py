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

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
_LAYOUT_SUFFIXES = {".json", ".fig", ".figma", ".sketch", ".xd", ".css", ".scss"}
_REFERENCE_SUFFIXES = {".pdf", ".md", ".txt"}
_REGION_TOKENS = ("hero", "header", "footer", "sidebar", "modal", "card", "banner")


class ImageUIContextAdapter:
    def __init__(self, *, token_budget: int):
        self.token_budget = token_budget

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
        artifacts = [self._artifact_from_candidate(item, changed_files, task_text) for item in prioritized_candidates]
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
            task=AdvisorTask(domain="image-ui", text=task_text, type=task_type),
            context=AdvisorContext(
                summary="image-ui task context",
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
                    domain="image-ui",
                    supported_artifact_kinds=["image", "layout", "reference"],
                    supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
                    supports_changed_artifacts=True,
                    supports_symbol_regions=True,
                )
            ],
        )

    def _prioritize_candidates(self, candidate_files: list[CandidateFile], changed_files: list[str]) -> list[CandidateFile]:
        changed = set(changed_files)
        prioritized = []
        for item in candidate_files:
            artifact_kind = self._artifact_kind(item.path)
            score = item.score
            if item.path in changed:
                score += 0.35
            if artifact_kind == "image":
                score += 0.25
            elif artifact_kind == "layout":
                score += 0.15
            prioritized.append(CandidateFile(path=item.path, reason=item.reason, score=score))
        prioritized.sort(key=lambda item: (-item.score, item.path))
        return prioritized

    def _artifact_from_candidate(self, item: CandidateFile, changed_files: list[str], task_text: str) -> AdvisorArtifact:
        metadata = {"source": "candidate_files", "changed": item.path in set(changed_files)}
        region_hint = self._region_hint(task_text)
        if region_hint is not None:
            metadata["region_hint"] = region_hint
        return AdvisorArtifact(
            kind=self._artifact_kind(item.path),
            locator=item.path,
            description=item.reason,
            metadata=metadata,
            score=item.score,
        )

    def _artifact_kind(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        parts = {part.lower() for part in Path(path).parts}
        if suffix in _IMAGE_SUFFIXES or {"mockups", "screens", "screenshots", "images"} & parts:
            return "image"
        if suffix in _LAYOUT_SUFFIXES or {"layouts", "ui"} & parts:
            return "layout"
        if suffix in _REFERENCE_SUFFIXES or {"references", "guides", "tokens"} & parts:
            return "reference"
        return "reference"

    def _artifact_counts(self, artifacts: list[AdvisorArtifact]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for artifact in artifacts:
            counts[artifact.kind] = counts.get(artifact.kind, 0) + 1
        return counts

    def _region_hint(self, task_text: str) -> str | None:
        text = task_text.lower()
        for token in _REGION_TOKENS:
            if token in text:
                return token
        return None

    def _modules_from_tree(self, file_tree: list[str]) -> list[str]:
        modules = []
        seen = set()
        for rel in file_tree:
            top = rel.split("/", 1)[0]
            if top not in seen:
                seen.add(top)
                modules.append(top)
        return modules[:10]
