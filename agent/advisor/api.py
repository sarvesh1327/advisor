from __future__ import annotations

from .gateway import AdvisorGateway, create_app
from .orchestration import AdvisorOrchestrator
from .schemas import AdvisorTaskRunResult
from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore
from .version import __version__


# These helpers are the stable API layer used by CLI, tests, and external callers.
def get_version() -> str:
    return __version__


def create_gateway(
    *,
    settings: AdvisorSettings | None = None,
    runtime=None,
    trace_store=None,
) -> AdvisorGateway:
    return AdvisorGateway(settings=settings, runtime=runtime, trace_store=trace_store)


def create_orchestrator(
    *,
    executor,
    verifiers: list | None = None,
    settings: AdvisorSettings | None = None,
    runtime=None,
    trace_store=None,
    router=None,
    enable_second_pass_review: bool = False,
) -> AdvisorOrchestrator:
    active_settings = settings or AdvisorSettings.load()
    active_trace_store = trace_store
    if active_trace_store is None:
        active_settings.ensure_dirs()
        active_trace_store = AdvisorTraceStore(active_settings.trace_db_path)
    return AdvisorOrchestrator(
        runtime=runtime,
        trace_store=active_trace_store,
        executor=executor,
        verifiers=verifiers,
        settings=active_settings,
        router=router,
        enable_second_pass_review=enable_second_pass_review,
    )


def run_task(
    *,
    task_text: str,
    repo_path: str,
    gateway: AdvisorGateway | None = None,
    branch: str | None = None,
    tool_limits: dict | None = None,
    acceptance_criteria: list[str] | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    task_type_hint: str | None = None,
    system_prompt: str | None = None,
    changed_files: list[str] | None = None,
) -> AdvisorTaskRunResult:
    # Create a default gateway lazily so callers can inject their own runtime/store in tests.
    active_gateway = gateway or create_gateway()
    return active_gateway.task_run(
        task_text=task_text,
        repo_path=repo_path,
        branch=branch,
        tool_limits=tool_limits,
        acceptance_criteria=acceptance_criteria,
        session_id=session_id,
        task_id=task_id,
        task_type_hint=task_type_hint,
        system_prompt=system_prompt,
        changed_files=changed_files,
    )


def create_http_app(*, settings: AdvisorSettings | None = None):
    # HTTP creation stays thin so gateway behavior remains the single execution path.
    return create_app(settings=settings)
