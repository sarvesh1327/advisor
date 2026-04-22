from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
