from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CandidateFile(BaseModel):
    path: str
    reason: str
    score: float = 0.0


class FailureSignal(BaseModel):
    kind: str
    file: str | None = None
    summary: str
    fix_hint: str | None = None


class RepoSummary(BaseModel):
    modules: list[str] = Field(default_factory=list)
    hotspots: list[str] = Field(default_factory=list)
    file_tree_slice: list[str] = Field(default_factory=list)


class RelevantFile(BaseModel):
    path: str
    why: str
    priority: int


class RelevantSymbol(BaseModel):
    name: str
    path: str
    why: str


class FocusTarget(BaseModel):
    kind: str
    locator: str
    rationale: str
    priority: int = 1


class ExecutorInjectionPolicy(BaseModel):
    strategy: Literal["prepend", "append"] = "prepend"
    format: Literal["plain_text"] = "plain_text"
    min_confidence: float = 0.0
    include_confidence_note: bool = True


class AdvisorTask(BaseModel):
    # Core task identity used across all domain adapters.
    domain: str = "coding"
    text: str
    type: str


class AdvisorContext(BaseModel):
    # Adapter-specific metadata stays here so the core packet remains generic.
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdvisorArtifact(BaseModel):
    # Artifacts are domain-neutral focus targets: files, screenshots, docs, etc.
    kind: str
    locator: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class AdvisorHistoryEntry(BaseModel):
    # History entries preserve prior signals without assuming coding-only failures.
    kind: str
    summary: str
    locator: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdvisorCapabilityDescriptor(BaseModel):
    # Adapters declare which packet features a runtime can safely depend on.
    domain: str
    supported_artifact_kinds: list[str] = Field(default_factory=list)
    supported_packet_fields: list[str] = Field(default_factory=list)
    supports_changed_artifacts: bool = False
    supports_symbol_regions: bool = False


class AdvisorTaskRequest(BaseModel):
    run_id: str | None = None
    task_text: str
    repo_path: str
    branch: str | None = None
    task_type_hint: str | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    tool_limits: dict[str, Any] = Field(default_factory=dict)


class AdvisorInputPacket(BaseModel):
    run_id: str
    task_text: str
    task_type: str
    repo: dict[str, Any]
    repo_summary: RepoSummary
    candidate_files: list[CandidateFile] = Field(default_factory=list)
    recent_failures: list[FailureSignal] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tool_limits: dict[str, Any] = Field(default_factory=dict)
    acceptance_criteria: list[str] = Field(default_factory=list)
    token_budget: int
    task: AdvisorTask | None = None
    context: AdvisorContext | None = None
    artifacts: list[AdvisorArtifact] = Field(default_factory=list)
    history: list[AdvisorHistoryEntry] = Field(default_factory=list)
    domain_capabilities: list[AdvisorCapabilityDescriptor] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_generic_fields(self) -> "AdvisorInputPacket":
        # Backfill the generic packet shape from legacy coding fields during migration.
        if self.task is None:
            self.task = AdvisorTask(domain="coding", text=self.task_text, type=self.task_type)
        if self.context is None:
            self.context = AdvisorContext(
                summary=f"{self.task.domain} task context",
                metadata={
                    "repo": self.repo,
                    "repo_summary": self.repo_summary.model_dump(),
                    "tool_limits": self.tool_limits,
                    "token_budget": self.token_budget,
                },
            )
        if not self.artifacts:
            self.artifacts = [
                AdvisorArtifact(
                    kind="file",
                    locator=item.path,
                    description=item.reason,
                    metadata={"source": "candidate_files"},
                    score=item.score,
                )
                for item in self.candidate_files
            ]
        if not self.history:
            self.history = [
                AdvisorHistoryEntry(
                    kind=item.kind,
                    summary=item.summary,
                    locator=item.file,
                    metadata={"fix_hint": item.fix_hint} if item.fix_hint else {},
                )
                for item in self.recent_failures
            ]
        if not self.domain_capabilities:
            self.domain_capabilities = [
                AdvisorCapabilityDescriptor(
                    domain=self.task.domain,
                    supported_artifact_kinds=sorted({item.kind for item in self.artifacts}) or ["file"],
                    supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
                    supports_changed_artifacts=True,
                    supports_symbol_regions=False,
                )
            ]
        return self


class AdviceBlock(BaseModel):
    task_type: str
    focus_targets: list[FocusTarget] = Field(default_factory=list)
    relevant_files: list[RelevantFile] = Field(default_factory=list)
    relevant_symbols: list[RelevantSymbol] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    likely_failure_modes: list[str] = Field(default_factory=list)
    recommended_plan: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str | None = None
    injection_policy: ExecutorInjectionPolicy = Field(default_factory=ExecutorInjectionPolicy)

    @model_validator(mode="after")
    def populate_generic_advice_fields(self) -> "AdviceBlock":
        # Keep generic focus targets canonical while compatibility fields remain populated.
        if not self.focus_targets:
            priority = 1
            self.focus_targets = [
                FocusTarget(
                    kind="file",
                    locator=item.path,
                    rationale=item.why,
                    priority=item.priority,
                )
                for item in self.relevant_files
            ]
            priority = max((item.priority for item in self.focus_targets), default=0) + 1
            self.focus_targets.extend(
                FocusTarget(
                    kind="symbol",
                    locator=f"{item.path}::{item.name}" if item.path else item.name,
                    rationale=item.why,
                    priority=priority + index,
                )
                for index, item in enumerate(self.relevant_symbols)
            )
        if not self.relevant_files:
            self.relevant_files = [
                RelevantFile(path=item.locator, why=item.rationale, priority=item.priority)
                for item in self.focus_targets
                if item.kind == "file"
            ]
        if not self.relevant_symbols:
            self.relevant_symbols = []
            for item in self.focus_targets:
                if item.kind != "symbol":
                    continue
                path, _, name = item.locator.partition("::")
                symbol_name = name or path
                symbol_path = path if name else ""
                self.relevant_symbols.append(
                    RelevantSymbol(name=symbol_name, path=symbol_path, why=item.rationale)
                )
        return self


class AdvisorOutcome(BaseModel):
    run_id: str
    status: Literal["success", "failure", "partial"]
    files_touched: list[str] = Field(default_factory=list)
    retries: int = 0
    tests_run: list[str] = Field(default_factory=list)
    review_verdict: str | None = None
    summary: str | None = None


class AdvisorTaskRunResult(BaseModel):
    run_id: str
    advisor_input_packet: AdvisorInputPacket
    advice_block: AdviceBlock
    model_version: str
    latency_ms: int
