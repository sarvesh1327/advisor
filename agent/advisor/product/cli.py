from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.learning.controller import AutonomousLearningController
from agent.advisor.learning.service import run_autonomous_learning_service
from agent.advisor.operators.operator_runtime import (
    OperatorJobQueue,
    RetentionEnforcer,
    build_deployment_profile,
    build_operator_snapshot,
    enqueue_forced_profile_eval,
    inspect_profile_checkpoints,
    run_operator_job,
)
from agent.advisor.product.api import create_gateway, create_http_app, get_version
from agent.advisor.product.hardening import (
    build_alert_summary,
    build_deployment_hardening_profile,
    build_phase8_validation_report,
    evaluate_release_gate,
    export_product_bundle,
    import_product_bundle,
)
from agent.advisor.training.training_runtime import CheckpointLifecycleManager

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

    operator_run_parser = subparsers.add_parser("operator-run-job", help="Run a queued operator job by id")
    operator_run_parser.add_argument("--job-id", required=True, help="Queued operator job id")
    operator_run_parser.set_defaults(handler=_handle_operator_run_job)

    operator_queue_status_parser = subparsers.add_parser("operator-queue-status", help="Print operator queue state JSON")
    operator_queue_status_parser.set_defaults(handler=_handle_operator_queue_status)

    operator_queue_pause_parser = subparsers.add_parser("operator-queue-pause", help="Pause the operator queue")
    operator_queue_pause_parser.add_argument("--reason", default=None, help="Optional pause reason")
    operator_queue_pause_parser.set_defaults(handler=_handle_operator_queue_pause)

    operator_queue_resume_parser = subparsers.add_parser("operator-queue-resume", help="Resume the operator queue")
    operator_queue_resume_parser.set_defaults(handler=_handle_operator_queue_resume)

    operator_checkpoints_parser = subparsers.add_parser("operator-checkpoints", help="Inspect checkpoints for a profile")
    operator_checkpoints_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    operator_checkpoints_parser.set_defaults(handler=_handle_operator_checkpoints)

    operator_force_eval_parser = subparsers.add_parser("operator-force-eval", help="Enqueue a forced profile eval job")
    operator_force_eval_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    operator_force_eval_parser.add_argument("--checkpoint-id", required=True, help="Candidate checkpoint id")
    operator_force_eval_parser.add_argument("--benchmark-manifests-path", required=True, help="Path to benchmark manifests JSON")
    operator_force_eval_parser.add_argument("--promotion-threshold", type=float, default=0.05, help="Promotion threshold")
    operator_force_eval_parser.set_defaults(handler=_handle_operator_force_eval)

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

    validation_gate_parser = subparsers.add_parser("validation-gate", help="Evaluate the Phase 8 final validation gate from local state")
    validation_gate_parser.add_argument("--required-profile", action="append", default=[], help="Repeatable required advisor profile id")
    validation_gate_parser.set_defaults(handler=_handle_validation_gate)

    learning_status_parser = subparsers.add_parser("learning-controller-status", help="Print autonomous learning controller state JSON")
    learning_status_parser.set_defaults(handler=_handle_learning_controller_status)

    learning_pause_parser = subparsers.add_parser("learning-controller-pause", help="Pause the autonomous learning controller")
    learning_pause_parser.add_argument("--reason", default=None, help="Optional pause reason")
    learning_pause_parser.set_defaults(handler=_handle_learning_controller_pause)

    learning_resume_parser = subparsers.add_parser("learning-controller-resume", help="Resume the autonomous learning controller")
    learning_resume_parser.set_defaults(handler=_handle_learning_controller_resume)

    learning_readiness_parser = subparsers.add_parser("learning-readiness", help="Print autonomous learning readiness for a profile")
    learning_readiness_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    learning_readiness_parser.set_defaults(handler=_handle_learning_readiness)

    learning_profile_pause_parser = subparsers.add_parser("learning-profile-pause", help="Pause autonomous learning for one profile")
    learning_profile_pause_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    learning_profile_pause_parser.add_argument("--reason", default=None, help="Optional pause reason")
    learning_profile_pause_parser.set_defaults(handler=_handle_learning_profile_pause)

    learning_profile_resume_parser = subparsers.add_parser("learning-profile-resume", help="Resume autonomous learning for one profile")
    learning_profile_resume_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    learning_profile_resume_parser.set_defaults(handler=_handle_learning_profile_resume)

    learning_profile_reset_parser = subparsers.add_parser("learning-profile-reset-backoff", help="Reset autonomous learning backoff for one profile")
    learning_profile_reset_parser.add_argument("--advisor-profile-id", required=True, help="Advisor profile id")
    learning_profile_reset_parser.set_defaults(handler=_handle_learning_profile_reset_backoff)

    learning_tick_parser = subparsers.add_parser("learning-tick", help="Run one autonomous learning controller tick")
    learning_tick_parser.set_defaults(handler=_handle_learning_tick)

    learning_service_parser = subparsers.add_parser("learning-service", help="Run the autonomous learning service loop")
    learning_service_parser.add_argument("--max-ticks", type=int, default=1, help="Maximum controller ticks to run")
    learning_service_parser.set_defaults(handler=_handle_learning_service)

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


def _handle_operator_run_job(args) -> int:
    settings = AdvisorSettings.load()
    gateway = create_gateway(settings=settings)
    queue = OperatorJobQueue(_queue_path(settings))
    record = run_operator_job(
        queue,
        args.job_id,
        settings=settings,
        profile_registry=gateway.profile_registry,
    )
    print(json.dumps(record.model_dump(), ensure_ascii=False))
    return 0



def _handle_operator_queue_status(args) -> int:
    del args
    settings = AdvisorSettings.load()
    queue = OperatorJobQueue(_queue_path(settings))
    print(json.dumps(queue.queue_status(), ensure_ascii=False))
    return 0



def _handle_operator_queue_pause(args) -> int:
    settings = AdvisorSettings.load()
    queue = OperatorJobQueue(_queue_path(settings))
    print(json.dumps(queue.pause(reason=args.reason), ensure_ascii=False))
    return 0



def _handle_operator_queue_resume(args) -> int:
    del args
    settings = AdvisorSettings.load()
    queue = OperatorJobQueue(_queue_path(settings))
    print(json.dumps(queue.resume_queue(), ensure_ascii=False))
    return 0



def _handle_operator_checkpoints(args) -> int:
    settings = AdvisorSettings.load()
    lifecycle_manager = _build_lifecycle_manager(settings)
    payload = inspect_profile_checkpoints(lifecycle_manager, advisor_profile_id=args.advisor_profile_id)
    print(json.dumps(payload, ensure_ascii=False))
    return 0



def _handle_operator_force_eval(args) -> int:
    settings = AdvisorSettings.load()
    queue = OperatorJobQueue(_queue_path(settings))
    record = enqueue_forced_profile_eval(
        queue,
        advisor_profile_id=args.advisor_profile_id,
        candidate_checkpoint_id=args.checkpoint_id,
        benchmark_manifests=_load_json_list(args.benchmark_manifests_path),
        promotion_threshold=args.promotion_threshold,
    )
    print(json.dumps(record.model_dump(), ensure_ascii=False))
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



def _handle_validation_gate(args) -> int:
    settings = AdvisorSettings.load()
    report = build_phase8_validation_report(
        lifecycle_manager=_build_lifecycle_manager(settings),
        job_records=OperatorJobQueue(_queue_path(settings)).list_jobs(),
        required_profiles=args.required_profile,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0



def _handle_learning_controller_status(args) -> int:
    del args
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.controller_status(), ensure_ascii=False))
    return 0



def _handle_learning_controller_pause(args) -> int:
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.pause_controller(reason=args.reason), ensure_ascii=False))
    return 0



def _handle_learning_controller_resume(args) -> int:
    del args
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.resume_controller(), ensure_ascii=False))
    return 0



def _handle_learning_readiness(args) -> int:
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.readiness_report(args.advisor_profile_id), ensure_ascii=False))
    return 0



def _handle_learning_profile_pause(args) -> int:
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.pause_profile(args.advisor_profile_id, reason=args.reason), ensure_ascii=False))
    return 0



def _handle_learning_profile_resume(args) -> int:
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.resume_profile(args.advisor_profile_id), ensure_ascii=False))
    return 0



def _handle_learning_profile_reset_backoff(args) -> int:
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.reset_profile_backoff(args.advisor_profile_id), ensure_ascii=False))
    return 0



def _handle_learning_tick(args) -> int:
    del args
    settings = AdvisorSettings.load()
    controller = AutonomousLearningController(settings=settings)
    print(json.dumps(controller.tick(), ensure_ascii=False))
    return 0



def _handle_learning_service(args) -> int:
    settings = AdvisorSettings.load()
    result = run_autonomous_learning_service(settings=settings, max_ticks=args.max_ticks, sleep_fn=lambda _: None)
    print(json.dumps(result, ensure_ascii=False))
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



def _queue_path(settings: AdvisorSettings) -> Path:
    return Path(settings.trace_db_path).expanduser().parent / "operator" / "jobs.json"



def _build_lifecycle_manager(settings: AdvisorSettings) -> CheckpointLifecycleManager:
    return CheckpointLifecycleManager(Path(settings.trace_db_path).expanduser().parent / "artifacts")



def _load_json_list(path: str) -> list[dict]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list at {path}")
    return payload



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
