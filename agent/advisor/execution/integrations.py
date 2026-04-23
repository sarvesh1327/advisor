from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket
from agent.advisor.execution.orchestration import (
    BuildTestVerifier,
    CodingAgentExecutor,
    DomainWorkerExecutor,
    ExecutorRequest,
    ExecutorRunResult,
    FrontierChatExecutor,
    HumanReviewVerifier,
    RubricVerifier,
    ScreenshotComparisonVerifier,
    VerifierResult,
)


class FrontierHTTPExecutor(FrontierChatExecutor):
    def __init__(self, *, name: str, endpoint_url: str, headers: dict[str, str] | None = None, timeout_seconds: int = 30):
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        super().__init__(
            name=name,
            metadata={"endpoint_url": endpoint_url, "transport": "http"},
            execute_fn=self._execute_http,
        )

    def execute_request(
        self,
        packet: AdvisorInputPacket,
        *,
        advice: AdviceBlock | None = None,
        rendered_advice: str | None = None,
    ) -> ExecutorRunResult:
        request = _build_executor_request(packet, advice=advice, rendered_advice=rendered_advice)
        return self.execute(request)

    def _execute_http(self, request: ExecutorRequest) -> ExecutorRunResult:
        payload = {
            "run_id": request.run_id,
            "packet": request.packet.model_dump(),
            "advice": request.advice.model_dump(),
            "rendered_advice": request.rendered_advice,
            "routing_decision": request.routing_decision.model_dump(),
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(self.endpoint_url, data=encoded, method="POST")
        http_request.add_header("content-type", "application/json")
        for key, value in self.headers.items():
            http_request.add_header(key, value)
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            return ExecutorRunResult(status="failure", summary=f"HTTP {exc.code}", output=detail)
        return _normalize_executor_result(json.loads(body))


class _SubprocessExecutorMixin:
    def __init__(self, *, command: list[str], cwd: str | None = None, timeout_seconds: int = 60):
        self.command = command
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

    def _run_subprocess(self, request: ExecutorRequest) -> ExecutorRunResult:
        payload = {
            "run_id": request.run_id,
            "packet": request.packet.model_dump(),
            "advice": request.advice.model_dump(),
            "rendered_advice": request.rendered_advice,
            "routing_decision": request.routing_decision.model_dump(),
        }
        candidate_cwd = self.cwd or request.packet.repo.get("path") or None
        cwd = candidate_cwd if candidate_cwd and Path(candidate_cwd).exists() else None
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload),
            capture_output=True,
            cwd=cwd,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            return ExecutorRunResult(
                status="failure",
                summary=f"executor exited with code {completed.returncode}",
                output=completed.stderr or completed.stdout,
                metadata={"returncode": completed.returncode},
            )
        return _normalize_executor_result(json.loads(completed.stdout or "{}"))


class CodingAgentSubprocessExecutor(_SubprocessExecutorMixin, CodingAgentExecutor):
    def __init__(self, *, name: str, command: list[str], cwd: str | None = None, timeout_seconds: int = 60):
        _SubprocessExecutorMixin.__init__(self, command=command, cwd=cwd, timeout_seconds=timeout_seconds)
        CodingAgentExecutor.__init__(
            self,
            name=name,
            metadata={"command": command, "transport": "subprocess"},
            execute_fn=self._run_subprocess,
        )

    def execute_request(
        self,
        packet: AdvisorInputPacket,
        *,
        advice: AdviceBlock | None = None,
        rendered_advice: str | None = None,
    ) -> ExecutorRunResult:
        return self.execute(_build_executor_request(packet, advice=advice, rendered_advice=rendered_advice))


class DomainWorkerSubprocessExecutor(_SubprocessExecutorMixin, DomainWorkerExecutor):
    def __init__(self, *, name: str, command: list[str], cwd: str | None = None, timeout_seconds: int = 60):
        _SubprocessExecutorMixin.__init__(self, command=command, cwd=cwd, timeout_seconds=timeout_seconds)
        DomainWorkerExecutor.__init__(
            self,
            name=name,
            metadata={"command": command, "transport": "subprocess"},
            execute_fn=self._run_subprocess,
        )

    def execute_request(
        self,
        packet: AdvisorInputPacket,
        *,
        advice: AdviceBlock | None = None,
        rendered_advice: str | None = None,
    ) -> ExecutorRunResult:
        return self.execute(_build_executor_request(packet, advice=advice, rendered_advice=rendered_advice))


class BuildTestCommandVerifier(BuildTestVerifier):
    def __init__(self, *, name: str, command: list[str], timeout_seconds: int = 60):
        self.command = command
        self.timeout_seconds = timeout_seconds
        super().__init__(
            name=name,
            metadata={"command": command, "transport": "subprocess"},
            verify_fn=self._verify_command,
        )

    def verify_request(self, packet: AdvisorInputPacket, executor_result: ExecutorRunResult) -> VerifierResult:
        return self.verify(_build_executor_request(packet), executor_result)

    def _verify_command(self, request: ExecutorRequest, result: ExecutorRunResult) -> VerifierResult:
        completed = subprocess.run(
            self.command,
            capture_output=True,
            cwd=request.packet.repo.get("path") or None,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        status = "pass" if completed.returncode == 0 else "fail"
        summary = "command passed" if completed.returncode == 0 else f"command failed with code {completed.returncode}"
        return VerifierResult(
            status=status,
            summary=summary,
            constraint_violations=[] if completed.returncode == 0 else [summary],
            metadata={"stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode},
        )


class RubricTextVerifier(RubricVerifier):
    def __init__(self, *, name: str, required_phrases: list[str]):
        self.required_phrases = required_phrases
        super().__init__(
            name=name,
            metadata={"required_phrases": required_phrases},
            verify_fn=self._verify_rubric,
        )

    def verify_request(self, packet: AdvisorInputPacket, executor_result: ExecutorRunResult) -> VerifierResult:
        return self.verify(_build_executor_request(packet), executor_result)

    def _verify_rubric(self, request: ExecutorRequest, result: ExecutorRunResult) -> VerifierResult:
        text = "\n".join(filter(None, [result.summary, result.output]))
        missing = [phrase for phrase in self.required_phrases if phrase not in text]
        return VerifierResult(
            status="pass" if not missing else "fail",
            summary="rubric satisfied" if not missing else f"missing rubric phrases: {', '.join(missing)}",
            constraint_violations=[f"missing rubric phrase: {phrase}" for phrase in missing],
            metadata={"required_phrases": self.required_phrases},
        )


class ScreenshotHashVerifier(ScreenshotComparisonVerifier):
    def __init__(self, *, name: str):
        super().__init__(name=name, metadata={"strategy": "sha256"}, verify_fn=self._verify_placeholder)

    def verify_artifacts(self, *, run_id: str, expected_path: str | Path, actual_path: str | Path) -> VerifierResult:
        expected = Path(expected_path).read_bytes()
        actual = Path(actual_path).read_bytes()
        expected_hash = hashlib.sha256(expected).hexdigest()
        actual_hash = hashlib.sha256(actual).hexdigest()
        match = expected_hash == actual_hash
        return VerifierResult(
            status="pass" if match else "fail",
            summary="screenshots match" if match else "screenshots differ",
            constraint_violations=[] if match else [f"screenshot mismatch for run {run_id}"],
            metadata={"expected_hash": expected_hash, "actual_hash": actual_hash},
        )

    def _verify_placeholder(self, request: ExecutorRequest, result: ExecutorRunResult) -> VerifierResult:
        return VerifierResult(status="warn", summary="use verify_artifacts() for screenshot checks")


class HumanReviewFileVerifier(HumanReviewVerifier):
    def __init__(self, *, name: str, review_file: str | Path):
        self.review_file = Path(review_file)
        super().__init__(
            name=name,
            metadata={"review_file": str(self.review_file)},
            verify_fn=self._verify_review,
        )

    def verify_request(self, packet: AdvisorInputPacket, executor_result: ExecutorRunResult) -> VerifierResult:
        return self.verify(_build_executor_request(packet), executor_result)

    def _verify_review(self, request: ExecutorRequest, result: ExecutorRunResult) -> VerifierResult:
        reviews = json.loads(self.review_file.read_text(encoding="utf-8")) if self.review_file.exists() else {}
        record = reviews.get(request.run_id, {})
        status = record.get("status", "warn")
        return VerifierResult(
            status=status,
            summary=record.get("summary", "human review pending"),
            constraint_violations=record.get("constraint_violations", []),
            metadata={"reviewer": record.get("reviewer")},
        )


class IntegrationRegistry:
    def create_executor(self, config: dict[str, Any]):
        kind = config["kind"]
        if kind == "frontier_http":
            return FrontierHTTPExecutor(
                name=config["name"],
                endpoint_url=config["endpoint_url"],
                headers=config.get("headers"),
                timeout_seconds=config.get("timeout_seconds", 30),
            )
        if kind == "coding_agent_subprocess":
            return CodingAgentSubprocessExecutor(
                name=config["name"],
                command=list(config["command"]),
                cwd=config.get("cwd"),
                timeout_seconds=config.get("timeout_seconds", 60),
            )
        if kind == "domain_worker_subprocess":
            return DomainWorkerSubprocessExecutor(
                name=config["name"],
                command=list(config["command"]),
                cwd=config.get("cwd"),
                timeout_seconds=config.get("timeout_seconds", 60),
            )
        raise ValueError(f"unknown executor integration kind '{kind}'")

    def create_verifier(self, config: dict[str, Any]):
        kind = config["kind"]
        if kind == "build_test_command":
            return BuildTestCommandVerifier(
                name=config["name"],
                command=list(config["command"]),
                timeout_seconds=config.get("timeout_seconds", 60),
            )
        if kind == "rubric_text":
            return RubricTextVerifier(name=config["name"], required_phrases=list(config["required_phrases"]))
        if kind == "screenshot_hash":
            return ScreenshotHashVerifier(name=config["name"])
        if kind == "human_review_file":
            return HumanReviewFileVerifier(name=config["name"], review_file=config["review_file"])
        raise ValueError(f"unknown verifier integration kind '{kind}'")


def _normalize_executor_result(payload: dict[str, Any]) -> ExecutorRunResult:
    return ExecutorRunResult(
        status=payload.get("status", "failure"),
        summary=payload.get("summary"),
        output=payload.get("output"),
        files_touched=list(payload.get("files_touched", [])),
        tests_run=list(payload.get("tests_run", [])),
        metadata=dict(payload.get("metadata", {})),
        artifacts=list(payload.get("artifacts", [])),
        retries=int(payload.get("retries", 0)),
    )


def _build_executor_request(
    packet: AdvisorInputPacket,
    *,
    advice: AdviceBlock | None = None,
    rendered_advice: str | None = None,
) -> ExecutorRequest:
    advice = advice or AdviceBlock(task_type=packet.task_type, confidence=0.0)
    return ExecutorRequest(
        run_id=packet.run_id,
        packet=packet,
        advice=advice,
        rendered_advice=rendered_advice,
        routing_decision={
            "arm": "advisor" if rendered_advice else "baseline",
            "advisor_fraction": 1.0 if rendered_advice else 0.0,
            "routing_key": packet.run_id,
            "bucket": 0.0,
        },
    )
