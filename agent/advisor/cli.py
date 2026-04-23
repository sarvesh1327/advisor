from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import create_gateway, create_http_app, get_version
from .hardening import (
    build_alert_summary,
    build_deployment_hardening_profile,
    evaluate_release_gate,
    export_product_bundle,
    import_product_bundle,
)
from .operator_runtime import OperatorJobQueue, RetentionEnforcer, build_deployment_profile, build_operator_snapshot
from .settings import AdvisorSettings

try:
    from uvicorn import run as uvicorn_run
except ImportError:
    uvicorn_run = None


# Keep the CLI surface minimal; all real behavior should flow through API/gateway layers.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="advisor", description="Advisor command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print Advisor version")
    version_parser.set_defaults(handler=_handle_version)

    run_parser = subparsers.add_parser("run", help="Run Advisor for a single task")
    run_parser.add_argument("--task-text", required=True, help="Task description")
    run_parser.add_argument("--repo-path", required=True, help="Path to the target repo")
    run_parser.add_argument("--branch", default=None, help="Optional git branch hint")
    run_parser.add_argument("--task-type-hint", default=None, help="Optional task type override")
    run_parser.add_argument("--system-prompt", default=None, help="Optional advisor system prompt override")
    run_parser.add_argument("--advisor-profile-id", default=None, help="Optional advisor profile override")
    run_parser.add_argument("--acceptance-criterion", action="append", default=[], help="Repeatable acceptance criteria")
    run_parser.add_argument("--tool-limit", action="append", default=[], help="Repeatable tool limit in key=value form")
    run_parser.set_defaults(handler=_handle_run)

    serve_parser = subparsers.add_parser("serve", help="Run the Advisor HTTP server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.set_defaults(handler=_handle_serve)

    operator_parser = subparsers.add_parser("operator-overview", help="Print operator overview JSON")
    operator_parser.set_defaults(handler=_handle_operator_overview)

    retention_parser = subparsers.add_parser("retention-enforce", help="Archive and prune retained runs/events")
    retention_parser.set_defaults(handler=_handle_retention_enforce)

    deployment_parser = subparsers.add_parser("deployment-profile", help="Print deployment profile guidance")
    deployment_parser.add_argument("--mode", choices=["single_tenant", "hosted"], default=None, help="Deployment mode override")
    deployment_parser.set_defaults(handler=_handle_deployment_profile)

    hardening_parser = subparsers.add_parser("hardening-profile", help="Print finished-product hardening profile JSON")
    hardening_parser.add_argument("--mode", choices=["single_tenant", "hosted"], default="single_tenant", help="Hardening mode")
    hardening_parser.set_defaults(handler=_handle_hardening_profile)

    release_gate_parser = subparsers.add_parser("release-gate", help="Evaluate a Phase 16 results report against release thresholds")
    release_gate_parser.add_argument("--report-path", required=True, help="Path to Phase 16 results report JSON")
    release_gate_parser.set_defaults(handler=_handle_release_gate)

    export_bundle_parser = subparsers.add_parser("export-bundle", help="Export product state bundle")
    export_bundle_parser.add_argument("--output-dir", required=True, help="Destination bundle directory")
    export_bundle_parser.set_defaults(handler=_handle_export_bundle)

    import_bundle_parser = subparsers.add_parser("import-bundle", help="Import previously exported product state bundle")
    import_bundle_parser.add_argument("--bundle-path", required=True, help="Source bundle directory")
    import_bundle_parser.add_argument("--target-root", required=True, help="Destination restore directory")
    import_bundle_parser.set_defaults(handler=_handle_import_bundle)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


def _handle_version(args) -> int:
    del args
    print(get_version())
    return 0


def _handle_run(args) -> int:
    # Emit machine-readable JSON so the CLI can feed larger agent pipelines.
    gateway = create_gateway()
    result = gateway.task_run(
        task_text=args.task_text,
        repo_path=args.repo_path,
        branch=args.branch,
        tool_limits=_parse_tool_limits(args.tool_limit),
        acceptance_criteria=args.acceptance_criterion,
        task_type_hint=args.task_type_hint,
        system_prompt=args.system_prompt,
        advisor_profile_id=args.advisor_profile_id,
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False))
    return 0


def _handle_serve(args) -> int:
    if uvicorn_run is None:
        raise RuntimeError("uvicorn is not installed. Install the advisor runtime extras to use 'advisor serve'.")
    app = create_http_app()
    uvicorn_run(app, host=args.host, port=args.port)
    return 0


def _handle_operator_overview(args) -> int:
    del args
    settings = AdvisorSettings.load()
    gateway = create_gateway(settings=settings)
    deployment = build_deployment_profile(
        settings=settings,
        mode="hosted" if settings.hosted_mode else "single_tenant",
    )
    queue = OperatorJobQueue(Path(settings.trace_db_path).expanduser().parent / "operator" / "jobs.json")
    snapshot = build_operator_snapshot(
        store=gateway.trace_store,
        settings=settings,
        deployment=deployment,
        job_records=queue.list_jobs(),
    )
    print(json.dumps(snapshot, ensure_ascii=False))
    return 0


def _handle_retention_enforce(args) -> int:
    del args
    settings = AdvisorSettings.load()
    gateway = create_gateway(settings=settings)
    report = RetentionEnforcer(store=gateway.trace_store, settings=settings).enforce()
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _handle_deployment_profile(args) -> int:
    settings = AdvisorSettings.load()
    profile = build_deployment_profile(
        settings=settings,
        mode=args.mode or ("hosted" if settings.hosted_mode else "single_tenant"),
    )
    print(json.dumps(profile.model_dump(), ensure_ascii=False))
    return 0


def _handle_hardening_profile(args) -> int:
    settings = AdvisorSettings.load()
    state_root = Path(settings.trace_db_path).expanduser().parent
    profile = build_deployment_hardening_profile(mode=args.mode, state_root=state_root)
    print(json.dumps(profile.model_dump(), ensure_ascii=False))
    return 0


def _handle_release_gate(args) -> int:
    report = json.loads(Path(args.report_path).expanduser().read_text(encoding="utf-8"))
    verdict = evaluate_release_gate(report)
    payload = {
        "verdict": verdict,
        "alerts": build_alert_summary(verdict),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _handle_export_bundle(args) -> int:
    settings = AdvisorSettings.load()
    bundle_path = export_product_bundle(output_dir=args.output_dir, settings=settings)
    print(json.dumps({"bundle_path": bundle_path}, ensure_ascii=False))
    return 0


def _handle_import_bundle(args) -> int:
    restored_root = import_product_bundle(bundle_path=args.bundle_path, target_root=args.target_root)
    print(json.dumps({"restored_root": str(restored_root)}, ensure_ascii=False))
    return 0


def _parse_tool_limits(entries: list[str]) -> dict:
    result = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"invalid tool limit '{item}': expected key=value")
        key, raw_value = item.split("=", 1)
        result[key] = _coerce_scalar(raw_value)
    return result


def _coerce_scalar(value: str):
    # CLI inputs arrive as strings; coerce only the obvious scalar cases.
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
        return int(normalized)
    try:
        return float(normalized)
    except ValueError:
        return value
