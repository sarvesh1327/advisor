from __future__ import annotations

import argparse
import json

from .api import create_gateway, create_http_app, get_version

try:
    from uvicorn import run as uvicorn_run
except ImportError:
    uvicorn_run = None


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
    run_parser.add_argument("--acceptance-criterion", action="append", default=[], help="Repeatable acceptance criteria")
    run_parser.add_argument("--tool-limit", action="append", default=[], help="Repeatable tool limit in key=value form")
    run_parser.set_defaults(handler=_handle_run)

    serve_parser = subparsers.add_parser("serve", help="Run the Advisor HTTP server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.set_defaults(handler=_handle_serve)

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
    gateway = create_gateway()
    result = gateway.task_run(
        task_text=args.task_text,
        repo_path=args.repo_path,
        branch=args.branch,
        tool_limits=_parse_tool_limits(args.tool_limit),
        acceptance_criteria=args.acceptance_criterion,
        task_type_hint=args.task_type_hint,
        system_prompt=args.system_prompt,
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False))
    return 0


def _handle_serve(args) -> int:
    if uvicorn_run is None:
        raise RuntimeError("uvicorn is not installed. Install the advisor runtime extras to use 'advisor serve'.")
    app = create_http_app()
    uvicorn_run(app, host=args.host, port=args.port)
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
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return normalized == "true"
    if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
        return int(normalized)
    try:
        return float(normalized)
    except ValueError:
        return value
