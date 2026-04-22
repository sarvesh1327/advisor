from __future__ import annotations

import time
import uuid
from hashlib import sha256
from typing import Any

from .context_builder import ContextBuilder
from .runtime_mlx import MLXAdvisorRuntime
from .schemas import AdvisorTaskRequest, AdvisorTaskRunResult
from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore
from .validator import AdviceValidator

try:
    from fastapi import FastAPI
except ImportError:
    FastAPI = None


class AdvisorGateway:
    def __init__(self, settings: AdvisorSettings | None = None, runtime: Any | None = None, trace_store: AdvisorTraceStore | None = None):
        self.settings = settings or AdvisorSettings.from_env()
        self.settings.ensure_dirs()
        self.trace_store = trace_store or AdvisorTraceStore(self.settings.trace_db_path)
        self.context_builder = ContextBuilder(
            self.trace_store,
            max_tree_entries=self.settings.max_tree_entries,
            max_candidate_files=self.settings.max_context_files,
            max_failures=self.settings.max_failures,
            token_budget=self.settings.token_budget,
        )
        self.runtime = runtime or MLXAdvisorRuntime(self.settings)
        self.validator = AdviceValidator()

    def task_run(
        self,
        *,
        task_text: str,
        repo_path: str,
        branch: str | None = None,
        tool_limits: dict | None = None,
        acceptance_criteria: list[str] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        task_type_hint: str | None = None,
    ) -> AdvisorTaskRunResult:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        packet = self.context_builder.build(
            task_text=task_text,
            repo_path=repo_path,
            tool_limits=tool_limits or {},
            acceptance_criteria=acceptance_criteria or [],
            run_id=run_id,
            branch=branch,
            task_type_hint=task_type_hint,
        )
        packet.repo["session_id"] = session_id
        packet.repo["task_id"] = task_id
        started = time.perf_counter()
        advice = self.runtime.generate_advice(packet)
        latency_ms = int((time.perf_counter() - started) * 1000)
        safe_advice = self.validator.validate(advice)
        prompt_hash = sha256((packet.task_text + self.settings.model_version).encode()).hexdigest()
        self.trace_store.record_task_run(
            packet,
            safe_advice,
            advisor_model=self.settings.model_version,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            validated=True,
        )
        return AdvisorTaskRunResult(
            run_id=run_id,
            advisor_input_packet=packet,
            advice_block=safe_advice,
            model_version=self.settings.model_version,
            latency_ms=latency_ms,
        )


def create_app(settings: AdvisorSettings | None = None):
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Install the web or advisor extras to create the HTTP app.")

    gateway = AdvisorGateway(settings=settings)
    app = FastAPI(title="Advisor", version="0.1.0")

    @app.post("/v1/advisor/task-run", response_model=AdvisorTaskRunResult)
    def task_run(req: AdvisorTaskRequest):
        return gateway.task_run(
            task_text=req.task_text,
            repo_path=req.repo_path,
            branch=req.branch,
            tool_limits=req.tool_limits,
            acceptance_criteria=req.acceptance_criteria,
            session_id=req.session_id,
            task_id=req.task_id,
            task_type_hint=req.task_type_hint,
        )

    return app
