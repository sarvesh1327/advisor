from __future__ import annotations

import inspect
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

from .context_builder import ContextBuilder
from .injector import render_advice_for_user_context
from .operator_runtime import (
    OperatorJobQueue,
    OperatorJobRequest,
    RetentionEnforcer,
    build_deployment_profile,
    build_operator_snapshot,
)
from .profiles import AdvisorProfile, AdvisorProfileRegistry
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
        self.profile_registry = self._load_profile_registry()
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

    def _load_profile_registry(self) -> AdvisorProfileRegistry:
        profiles_path = Path(self.settings.advisor_profiles_path).expanduser()
        if profiles_path.exists():
            return AdvisorProfileRegistry.from_toml(profiles_path)
        # Fall back to a synthetic one-profile registry so existing local installs stay bootable.
        return AdvisorProfileRegistry(
            default_profile_id=self.settings.advisor_profile_id,
            profiles={
                self.settings.advisor_profile_id: AdvisorProfile(
                    profile_id=self.settings.advisor_profile_id,
                    domain="coding",
                    description="Fallback profile created from runtime settings.",
                )
            },
        )

    def _resolve_profile_id(self, advisor_profile_id: str | None) -> str:
        return self.profile_registry.resolve(advisor_profile_id).profile_id

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
        advisor_profile_id: str | None = None,
        changed_files: list[str] | None = None,
    ) -> AdvisorTaskRunResult:
        resolved_profile_id = self._resolve_profile_id(advisor_profile_id)
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
            advisor_profile_id=resolved_profile_id,
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
            advisor_profile_id=resolved_profile_id,
            model_version=self.settings.model_version,
            latency_ms=latency_ms,
        )


def create_app(settings: AdvisorSettings | None = None, runtime: Any | None = None):
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed. Install the web or advisor extras to create the HTTP app."
        )

    gateway = AdvisorGateway(settings=settings, runtime=runtime)
    active_settings = gateway.settings
    operator_queue = OperatorJobQueue(Path(active_settings.trace_db_path).expanduser().parent / "operator" / "jobs.json")
    deployment = build_deployment_profile(
        settings=active_settings,
        mode="hosted" if active_settings.hosted_mode else "single_tenant",
    )
    retention = RetentionEnforcer(store=gateway.trace_store, settings=active_settings)
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
            advisor_profile_id=req.advisor_profile_id,
            changed_files=req.changed_files,
        )

    @app.get("/v1/operator/overview")
    def operator_overview():
        return build_operator_snapshot(
            store=gateway.trace_store,
            settings=active_settings,
            deployment=deployment,
            job_records=operator_queue.list_jobs(),
        )

    @app.get("/v1/operator/runs/{run_id}")
    def operator_run_details(run_id: str):
        run = gateway.trace_store.get_run(run_id)
        if run is None:
            return {"run": None, "lineage": None}
        return {"run": run, "lineage": gateway.trace_store.get_lineage(run_id)}

    @app.get("/v1/operator/jobs")
    def operator_jobs():
        return [item.model_dump() for item in operator_queue.list_jobs()]

    @app.post("/v1/operator/jobs")
    def operator_enqueue_job(req: OperatorJobRequest):
        return operator_queue.enqueue_job(
            job_type=req.job_type,
            payload=req.payload,
            resume_token=req.resume_token,
        ).model_dump()

    @app.post("/v1/operator/jobs/{job_id}/resume")
    def operator_resume_job(job_id: str):
        resumed = [item for item in operator_queue.resume_incomplete_jobs() if item.job_id == job_id]
        if resumed:
            return resumed[0].model_dump()
        existing = [item for item in operator_queue.list_jobs() if item.job_id == job_id]
        return existing[0].model_dump() if existing else {"job_id": job_id, "status": "missing"}

    @app.post("/v1/operator/retention/enforce")
    def operator_retention_enforce():
        return retention.enforce()

    return app
