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
        return self


class AdviceBlock(BaseModel):
    task_type: str
    relevant_files: list[RelevantFile] = Field(default_factory=list)
    relevant_symbols: list[RelevantSymbol] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    likely_failure_modes: list[str] = Field(default_factory=list)
    recommended_plan: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str | None = None


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
