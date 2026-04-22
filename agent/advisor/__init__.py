"""Public package surface for the standalone Advisor product."""

from .api import create_gateway, create_http_app, create_orchestrator, get_version, run_task
from .gateway import AdvisorGateway
from .observability import LiveMetricsSnapshot, RunEventLogger, build_audit_report, export_live_metrics, redact_packet
from .orchestration import (
    AdvisorOrchestrator,
    BuildTestVerifier,
    CodingAgentExecutor,
    DeterministicABRouter,
    DomainWorkerExecutor,
    FrontierChatExecutor,
    HumanReviewVerifier,
    RubricVerifier,
    ScreenshotComparisonVerifier,
)
from .schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTaskRequest,
    AdvisorTaskRunResult,
)
from .settings import AdvisorSettings
from .version import __version__

# Keep __all__ narrow so external callers depend on stable entrypoints only.
__all__ = [
    "__version__",
    "AdviceBlock",
    "AdvisorGateway",
    "AdvisorInputPacket",
    "AdvisorOrchestrator",
    "AdvisorOutcome",
    "AdvisorSettings",
    "AdvisorTaskRequest",
    "AdvisorTaskRunResult",
    "BuildTestVerifier",
    "CodingAgentExecutor",
    "DeterministicABRouter",
    "DomainWorkerExecutor",
    "FrontierChatExecutor",
    "HumanReviewVerifier",
    "LiveMetricsSnapshot",
    "RubricVerifier",
    "RunEventLogger",
    "ScreenshotComparisonVerifier",
    "build_audit_report",
    "create_gateway",
    "create_http_app",
    "create_orchestrator",
    "export_live_metrics",
    "get_version",
    "redact_packet",
    "run_task",
 ]
