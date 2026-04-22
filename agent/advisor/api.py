from __future__ import annotations

from .gateway import AdvisorGateway, create_app
from .schemas import AdvisorTaskRunResult
from .settings import AdvisorSettings
from .version import __version__


def get_version() -> str:
    return __version__


def create_gateway(
    *,
    settings: AdvisorSettings | None = None,
    runtime=None,
    trace_store=None,
) -> AdvisorGateway:
    return AdvisorGateway(settings=settings, runtime=runtime, trace_store=trace_store)


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
) -> AdvisorTaskRunResult:
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
    )


def create_http_app(*, settings: AdvisorSettings | None = None):
    return create_app(settings=settings)
