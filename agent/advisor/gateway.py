from __future__ import annotations

import inspect
import time
import uuid
from hashlib import sha256
from typing import Any

from .context_builder import ContextBuilder
from .injector import render_advice_for_user_context
from .runtime_mlx import MLXAdvisorRuntime
from .schemas import AdvisorTaskRequest, AdvisorTaskRunResult
from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore
from .validator import AdviceValidator
from .version import __version__

try:
    from fastapi import FastAPI
except ImportError:
    FastAPI = None


class AdvisorGateway:
    def __init__(
        self,
        settings: AdvisorSettings | None = None,
        runtime: Any | None = None,
        trace_store: AdvisorTraceStore | None = None,
    ):
        self.settings = settings or AdvisorSettings.load()
        self.settings.ensure_dirs()
        self.trace_store = trace_store or AdvisorTraceStore(self.settings.trace_db_path)
        # Gateway owns the canonical task-run pipeline used by CLI, API, and tests.
        self.context_builder = ContextBuilder(
            self.trace_store,
            max_tree_entries=self.settings.max_tree_entries,
            max_candidate_files=self.settings.max_context_files,
            max_failures=self.settings.max_failures,
            token_budget=self.settings.token_budget,
        )
        self.runtime = runtime or MLXAdvisorRuntime(self.settings)
        self.validator = AdviceValidator()
        warmup = getattr(self.runtime, "warmup", None)
        if self.settings.warm_load_on_start and callable(warmup):
            warmup()

    def system_health(self) -> dict:
        # Health is capability-oriented so callers can distinguish missing runtime vs. bad startup.
        runtime_caps_fn = getattr(self.runtime, "capabilities", None)
        runtime_health = runtime_caps_fn() if callable(runtime_caps_fn) else {
            "runtime": type(self.runtime).__name__,
            "available": True,
            "ready": True,
            "reason": None,
        }
        status = "ok" if runtime_health.get("available") else "degraded"
        return {
            "status": status,
            "version": __version__,
            "config": self.settings.health_payload(),
            "runtime": runtime_health,
        }

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
        system_prompt: str | None = None,
        changed_files: list[str] | None = None,
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
            changed_files=changed_files or [],
        )
        packet.repo["session_id"] = session_id
        packet.repo["task_id"] = task_id
        started = time.perf_counter()
        generate_advice = self.runtime.generate_advice
        # Older runtimes may not support prompt overrides yet; keep the interface backward-compatible.
        supports_system_prompt = "system_prompt" in inspect.signature(generate_advice).parameters
        if supports_system_prompt:
            advice = generate_advice(packet, system_prompt=system_prompt)
        else:
            advice = generate_advice(packet)
        latency_ms = int((time.perf_counter() - started) * 1000)
        safe_advice = self.validator.validate(advice)
        injected_rendered_advice = render_advice_for_user_context(safe_advice)
        # Prompt hash tracks the effective task/model pair used for the stored advice record.
        prompt_hash = sha256(
            (packet.task_text + self.settings.model_version).encode()
        ).hexdigest()
        self.trace_store.record_task_run(
            packet,
            safe_advice,
            advisor_model=self.settings.model_version,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            validated=True,
            injected_advice=safe_advice,
            injected_rendered_advice=injected_rendered_advice,
        )
        return AdvisorTaskRunResult(
            run_id=run_id,
            advisor_input_packet=packet,
            advice_block=safe_advice,
            model_version=self.settings.model_version,
            latency_ms=latency_ms,
        )


def create_app(settings: AdvisorSettings | None = None, runtime: Any | None = None):
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed. Install the web or advisor extras to create the HTTP app."
        )

    gateway = AdvisorGateway(settings=settings, runtime=runtime)
    app = FastAPI(title="Advisor", version=__version__)

    # Keep HTTP routes thin; the gateway should remain the single source of execution behavior.
    @app.get("/healthz")
    def healthz():
        return gateway.system_health()

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
            system_prompt=req.system_prompt,
            changed_files=req.changed_files,
        )

    return app
