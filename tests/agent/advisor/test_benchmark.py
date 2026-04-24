import json

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.benchmark import (
    BenchmarkRunManifest,
    BenchmarkSuite,
    build_benchmark_run_manifest,
    compare_benchmark_arms,
    freeze_benchmark_suite,
)
from agent.advisor.evaluation.eval_fixtures import EvalExpectation, EvalFixture, HumanReviewRubric
from agent.advisor.execution.orchestration import DeterministicABRouter, ExecutorRunResult, FrontierChatExecutor
from agent.advisor.product.api import create_orchestrator
from agent.advisor.storage.trace_store import AdvisorTraceStore


class StubRuntime:
    def generate_advice(self, packet, system_prompt=None):
        return AdviceBlock(
            task_type=packet.task_type,
            relevant_files=[{"path": "main.py", "why": "entrypoint", "priority": 1}],
            recommended_plan=["inspect main.py"],
            confidence=0.9,
        )


def _packet(run_id: str, task_text: str = "repair main flow"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text=task_text,
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False, "session_id": f"sess-{run_id}"},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests must pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["repair succeeds"],
        token_budget=900,
    )


def _fixture(run_id: str, fixture_id: str, domain: str = "coding"):
    packet = _packet(run_id)
    return EvalFixture(
        fixture_id=fixture_id,
        domain=domain,
        description=packet.task_text,
        input_packet=packet,
        expected_advice=EvalExpectation(
            focus_targets=["main.py"],
            anti_targets=["docs/brief.md"],
            required_plan_steps=["inspect main.py"],
            forbidden_plan_steps=["broad refactor"],
            expected_failure_modes=[],
        ),
        human_review_rubric=HumanReviewRubric(scale=[0, 1, 2, 3], criteria=["helpfulness"]),
    )


def test_freeze_benchmark_suite_assigns_reproducible_splits():
    fixtures = [
        _fixture("run-a", "coding-a", domain="coding"),
        _fixture("run-b", "research-a", domain="research"),
        _fixture("run-c", "domain-a", domain="image"),
    ]

    first = freeze_benchmark_suite("phase13-core", fixtures)
    second = freeze_benchmark_suite("phase13-core", fixtures)

    assert isinstance(first, BenchmarkSuite)
    assert [case.model_dump() for case in first.cases] == [case.model_dump() for case in second.cases]
    assert {case.split for case in first.cases} <= {"train_pool", "validation", "test"}


def test_build_benchmark_run_manifest_captures_canonical_replay_fields(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    orchestrator = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="success",
                summary="patched main.py",
                output="patched main.py",
                files_touched=["main.py"],
                tests_run=["pytest -q"],
                metadata={"provider": "stub"},
            ),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )
    packet = _packet("run-manifest")
    result = orchestrator.run(packet, advisor_profile_id="coding-default")
    fixture = _fixture("run-manifest", "coding-main")

    manifest = build_benchmark_run_manifest(
        store=store,
        run_id=result.run_id,
        fixture=fixture,
        split="validation",
    )

    assert isinstance(manifest, BenchmarkRunManifest)
    assert manifest.packet_hash == result.manifest.replay_inputs["packet_hash"]
    assert manifest.executor_config["name"] == "frontier-chat"
    assert manifest.verifier_set == []
    assert manifest.routing_arm == "advisor"
    assert manifest.reward_version == "coding-swe-efficiency-v1"
    assert manifest.advisor_profile_id == "coding-default"


def test_compare_benchmark_arms_is_reproducible_and_ablation_friendly():
    baseline = BenchmarkRunManifest(
        run_id="baseline-run",
        fixture_id="coding-main",
        domain="coding",
        split="validation",
        packet_hash="abc",
        executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
        verifier_set=["build-check"],
        routing_arm="baseline",
        advisor_profile_id=None,
        reward_version="phase8-v1",
        score={"overall_score": 0.45, "focus_target_recall": 0.5},
    )
    advisor = BenchmarkRunManifest(
        run_id="advisor-run",
        fixture_id="coding-main",
        domain="coding",
        split="validation",
        packet_hash="abc",
        executor_config={"name": "frontier-chat", "kind": "frontier_chat"},
        verifier_set=["build-check"],
        routing_arm="advisor",
        advisor_profile_id="coding-default",
        reward_version="phase8-v1",
        score={"overall_score": 0.9, "focus_target_recall": 1.0},
    )

    first = compare_benchmark_arms([advisor, baseline])
    second = compare_benchmark_arms([baseline, advisor])

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["arm_summary"]["advisor"]["mean_overall_score"] == 0.9
    assert first["arm_summary"]["baseline"]["mean_overall_score"] == 0.45
    assert first["deltas"]["advisor_minus_baseline"]["overall_score"] == 0.45
    assert first["by_split"]["validation"]["advisor"]["count"] == 1
    assert first["by_profile"]["coding-default"]["advisor"]["mean_overall_score"] == 0.9
    assert first["ablation_axes"] == {
        "domains": ["coding"],
        "executor_kinds": ["frontier_chat"],
        "reward_versions": ["phase8-v1"],
        "splits": ["validation"],
        "verifier_sets": ["build-check"],
        "advisor_profiles": ["coding-default"],
    }
