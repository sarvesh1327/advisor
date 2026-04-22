"""Advisor modules for local pre-execution steering."""

from .api import create_gateway, create_http_app, get_version, run_task
from .gateway import AdvisorGateway
from .schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTaskRequest,
    AdvisorTaskRunResult,
)
from .settings import AdvisorSettings
from .version import __version__

__all__ = [
    "__version__",
    "AdviceBlock",
    "AdvisorGateway",
    "AdvisorInputPacket",
    "AdvisorOutcome",
    "AdvisorSettings",
    "AdvisorTaskRequest",
    "AdvisorTaskRunResult",
    "create_gateway",
    "create_http_app",
    "get_version",
    "run_task",
]
