from __future__ import annotations

import time
from hashlib import sha256
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .injector import render_advice_for_user_context
from .reward_model import RewardWeights, compute_reward_label
from .schemas import (
    AdviceBlock,
    AdvisorArtifact,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorOutcome,
    RewardLabel,
)
from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore
from .validator import AdviceValidator


class RoutingDecision(BaseModel):
    arm: Literal["baseline", "advisor"]
    advisor_fraction: float
    routing_key: str
    bucket: float


class ExecutorDescriptor(BaseModel):
    name: str
    kind: Literal["frontier_chat", "coding_agent", "domain_worker"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierDescriptor(BaseModel):
    name: str
    kind: Literal["build_test", "screenshot", "rubric", "human_review"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutorRequest(BaseModel):
    run_id: str
    packet: AdvisorInputPacket
    advice: AdviceBlock
    rendered_advice: str | None = None
    routing_decision: RoutingDecision


class ExecutorRunResult(BaseModel):
    status: Literal["success", "failure", "partial"]
    summary: str | None = None
    output: str | None = None
    files_touched: list[str] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    retries: int = 0


class VerifierResult(BaseModel):
    status: Literal["pass", "fail", "warn"]
    summary: str | None = None
    constraint_violations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierRunRecord(BaseModel):
    descriptor: VerifierDescriptor
    result: VerifierResult


class RunManifest(BaseModel):
    run_id: str
    routing_decision: RoutingDecision
    executor: ExecutorDescriptor
    verifiers: list[VerifierDescriptor] = Field(default_factory=list)
    review_enabled: bool = False
    replay_inputs: dict[str, Any] = Field(default_factory=dict)


class RunLineage(BaseModel):
    run_id: str
    packet: AdvisorInputPacket
    primary_advice: AdviceBlock
    review_advice: AdviceBlock | None = None
    executor_result: ExecutorRunResult
    verifier_results: list[VerifierRunRecord] = Field(default_factory=list)
    outcome: AdvisorOutcome
    reward_label: RewardLabel


class LiveRunResult(BaseModel):
    run_id: str
    manifest: RunManifest
    lineage: RunLineage


class DeterministicABRouter:
    def __init__(self, advisor_fraction: float = 0.5):
        if not 0.0 <= advisor_fraction <= 1.0:
            raise ValueError("advisor_fraction must be between 0.0 and 1.0")
        self.advisor_fraction = advisor_fraction

    def choose(self, packet: AdvisorInputPacket) -> RoutingDecision:
        routing_key = str(
            packet.repo.get("session_id")
            or packet.repo.get("task_id")
            or packet.run_id
            or packet.task_text
        )
        bucket = int(sha256(routing_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        arm = "advisor" if bucket < self.advisor_fraction else "baseline"
        return RoutingDecision(
            arm=arm,
            advisor_fraction=self.advisor_fraction,
            routing_key=routing_key,
            bucket=round(bucket, 6),
        )


class CallableExecutor:
    def __init__(
        self,
        *,
        kind: Literal["frontier_chat", "coding_agent", "domain_worker"],
        name: str,
        execute_fn: Callable[[ExecutorRequest], ExecutorRunResult | dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        self.descriptor = ExecutorDescriptor(name=name, kind=kind, metadata=metadata or {})
        self._execute_fn = execute_fn

    def execute(self, request: ExecutorRequest) -> ExecutorRunResult:
        result = self._execute_fn(request)
        if isinstance(result, ExecutorRunResult):
            return result
        return ExecutorRunResult(**result)


class FrontierChatExecutor(CallableExecutor):
    def __init__(self, *, name: str, execute_fn: Callable[[ExecutorRequest], ExecutorRunResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="frontier_chat", name=name, execute_fn=execute_fn, metadata=metadata)


class CodingAgentExecutor(CallableExecutor):
    def __init__(self, *, name: str, execute_fn: Callable[[ExecutorRequest], ExecutorRunResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="coding_agent", name=name, execute_fn=execute_fn, metadata=metadata)


class DomainWorkerExecutor(CallableExecutor):
    def __init__(self, *, name: str, execute_fn: Callable[[ExecutorRequest], ExecutorRunResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="domain_worker", name=name, execute_fn=execute_fn, metadata=metadata)


class CallableVerifier:
    def __init__(
        self,
        *,
        kind: Literal["build_test", "screenshot", "rubric", "human_review"],
        name: str,
        verify_fn: Callable[[ExecutorRequest, ExecutorRunResult], VerifierResult | dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        self.descriptor = VerifierDescriptor(name=name, kind=kind, metadata=metadata or {})
        self._verify_fn = verify_fn

    def verify(self, request: ExecutorRequest, result: ExecutorRunResult) -> VerifierResult:
        verdict = self._verify_fn(request, result)
        if isinstance(verdict, VerifierResult):
            return verdict
        return VerifierResult(**verdict)


class BuildTestVerifier(CallableVerifier):
    def __init__(self, *, name: str, verify_fn: Callable[[ExecutorRequest, ExecutorRunResult], VerifierResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="build_test", name=name, verify_fn=verify_fn, metadata=metadata)


class ScreenshotComparisonVerifier(CallableVerifier):
    def __init__(self, *, name: str, verify_fn: Callable[[ExecutorRequest, ExecutorRunResult], VerifierResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="screenshot", name=name, verify_fn=verify_fn, metadata=metadata)


class RubricVerifier(CallableVerifier):
    def __init__(self, *, name: str, verify_fn: Callable[[ExecutorRequest, ExecutorRunResult], VerifierResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="rubric", name=name, verify_fn=verify_fn, metadata=metadata)


class HumanReviewVerifier(CallableVerifier):
    def __init__(self, *, name: str, verify_fn: Callable[[ExecutorRequest, ExecutorRunResult], VerifierResult | dict[str, Any]], metadata: dict[str, Any] | None = None):
        super().__init__(kind="human_review", name=name, verify_fn=verify_fn, metadata=metadata)


class AdvisorOrchestrator:
    def __init__(
        self,
        *,
        runtime,
        trace_store: AdvisorTraceStore,
        executor: CallableExecutor,
        verifiers: list[CallableVerifier] | None = None,
        settings: AdvisorSettings | None = None,
        router: DeterministicABRouter | None = None,
        enable_second_pass_review: bool = False,
    ):
        self.runtime = runtime
        self.trace_store = trace_store
        self.executor = executor
        self.verifiers = list(verifiers or [])
        self.settings = settings or AdvisorSettings.load()
        self.router = router or DeterministicABRouter(advisor_fraction=1.0)
        self.enable_second_pass_review = enable_second_pass_review
        self.validator = AdviceValidator()

    def run(self, packet: AdvisorInputPacket, *, system_prompt: str | None = None) -> LiveRunResult:
        primary_advice, latency_ms = self._generate_advice(packet, system_prompt=system_prompt)
        routing_decision = self.router.choose(packet)
        rendered_advice = render_advice_for_user_context(primary_advice) if routing_decision.arm == "advisor" else None
        prompt_hash = sha256((packet.task_text + self.settings.model_version).encode("utf-8")).hexdigest()
        self.trace_store.record_task_run(
            packet,
            primary_advice,
            advisor_model=self.settings.model_version,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            validated=True,
            injected_advice=primary_advice,
            injected_rendered_advice=rendered_advice,
        )
        executor_request = ExecutorRequest(
            run_id=packet.run_id,
            packet=packet,
            advice=primary_advice,
            rendered_advice=rendered_advice,
            routing_decision=routing_decision,
        )
        executor_result = self.executor.execute(executor_request)
        review_advice = self._review_executor_output(packet, executor_result, system_prompt=system_prompt)
        verifier_results = [
            VerifierRunRecord(descriptor=verifier.descriptor, result=verifier.verify(executor_request, executor_result))
            for verifier in self.verifiers
        ]
        outcome = self._build_outcome(packet.run_id, executor_result, verifier_results)
        self.trace_store.record_outcome(outcome)
        constraint_violations = [
            item
            for record in verifier_results
            for item in record.result.constraint_violations
            if item
        ]
        reward_label = compute_reward_label(
            packet,
            primary_advice,
            outcome,
            constraint_violations=constraint_violations,
            weights=RewardWeights(**self.settings.reward_weights().__dict__),
        )
        self.trace_store.record_reward_label(reward_label)
        manifest = RunManifest(
            run_id=packet.run_id,
            routing_decision=routing_decision,
            executor=self.executor.descriptor,
            verifiers=[verifier.descriptor for verifier in self.verifiers],
            review_enabled=self.enable_second_pass_review,
            replay_inputs={
                "system_prompt": system_prompt,
                "advisor_model": self.settings.model_version,
                "packet_hash": sha256(packet.model_dump_json().encode("utf-8")).hexdigest(),
            },
        )
        lineage = RunLineage(
            run_id=packet.run_id,
            packet=packet,
            primary_advice=primary_advice,
            review_advice=review_advice,
            executor_result=executor_result,
            verifier_results=verifier_results,
            outcome=outcome,
            reward_label=reward_label,
        )
        self.trace_store.record_lineage(packet.run_id, manifest, lineage)
        return LiveRunResult(run_id=packet.run_id, manifest=manifest, lineage=lineage)

    def _generate_advice(self, packet: AdvisorInputPacket, *, system_prompt: str | None) -> tuple[AdviceBlock, int]:
        started = time.perf_counter()
        advice = self.runtime.generate_advice(packet, system_prompt=system_prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return self.validator.validate(advice), latency_ms

    def _review_executor_output(
        self,
        packet: AdvisorInputPacket,
        executor_result: ExecutorRunResult,
        *,
        system_prompt: str | None,
    ) -> AdviceBlock | None:
        if not self.enable_second_pass_review:
            return None
        review_packet = packet.model_copy(deep=True)
        review_packet.task_text = f"Review executor output for run {packet.run_id}"
        review_packet.task_type = "review"
        if review_packet.task is not None:
            review_packet.task.text = review_packet.task_text
            review_packet.task.type = "review"
        review_packet.artifacts.append(
            AdvisorArtifact(
                kind="executor_output",
                locator=f"run:{packet.run_id}",
                description=executor_result.summary,
                metadata={
                    "output": executor_result.output,
                    "files_touched": executor_result.files_touched,
                    "tests_run": executor_result.tests_run,
                },
            )
        )
        review_packet.history.append(
            AdvisorHistoryEntry(
                kind="executor_result",
                summary=executor_result.summary or executor_result.status,
                locator=f"run:{packet.run_id}",
                metadata={"status": executor_result.status},
            )
        )
        review_advice, _ = self._generate_advice(review_packet, system_prompt=system_prompt)
        return review_advice

    def _build_outcome(
        self,
        run_id: str,
        executor_result: ExecutorRunResult,
        verifier_results: list[VerifierRunRecord],
    ) -> AdvisorOutcome:
        statuses = {record.result.status for record in verifier_results}
        if executor_result.status == "failure" or "fail" in statuses:
            status: Literal["success", "failure", "partial"] = "failure"
            review_verdict = "fail"
        elif executor_result.status == "partial" or "warn" in statuses:
            status = "partial"
            review_verdict = "warn"
        else:
            status = "success"
            review_verdict = "pass"
        summary_parts = [executor_result.summary] if executor_result.summary else []
        summary_parts.extend(record.result.summary for record in verifier_results if record.result.summary)
        return AdvisorOutcome(
            run_id=run_id,
            status=status,
            files_touched=executor_result.files_touched,
            retries=executor_result.retries,
            tests_run=executor_result.tests_run,
            review_verdict=review_verdict,
            summary=" | ".join(summary_parts) if summary_parts else None,
        )
