import json

from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.evaluation.benchmark import BenchmarkRunManifest
from agent.advisor.evaluation.results_pass import (
    build_failure_taxonomy,
    build_phase16_results_report,
    default_paper_divergences,
    summarize_ablation_results,
    summarize_canonical_study,
    summarize_provenance_coverage,
    summarize_transfer_results,
    write_phase16_results_report,
)
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


def _seed_store(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    orchestrator = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=FrontierChatExecutor(
            name="frontier-chat",
            execute_fn=lambda request: ExecutorRunResult(
                status="failure" if request.packet.run_id == "run-fail" else "success",
                summary="pytest timeout in main.py" if request.packet.run_id == "run-fail" else "patched main.py",
                output="patched main.py",
                files_touched=["main.py"],
                tests_run=["pytest -q"],
                metadata={"provider": "stub"},
            ),
        ),
        verifiers=[],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )
    orchestrator.run(_packet("run-good"))
    orchestrator.run(_packet("run-fail"))
    return store


def test_summarize_canonical_study_is_deterministic_and_reports_lift():
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

    first = summarize_canonical_study([advisor, baseline])
    second = summarize_canonical_study([baseline, advisor])

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["lift_summary"]["advisor_minus_baseline_overall_score"] == 0.45
    assert first["protocol"]["parity_rule"] == "same executor, same verifier set, advice injection only"
    assert first["by_profile"]["coding-default"]["advisor"]["count"] == 1


def test_summarize_ablation_and_transfer_results_group_variants_and_executors():
    ablations = summarize_ablation_results(
        [
            {"variant_id": "packet_field:history", "kind": "packet_field", "target": "history", "overall_score_delta": -0.08},
            {"variant_id": "advice_field:focus_targets", "kind": "advice_field", "target": "focus_targets", "overall_score_delta": -0.03},
            {"variant_id": "reward_component:human_usefulness", "kind": "reward_component", "target": "human_usefulness", "overall_score_delta": 0.02},
        ]
    )
    transfer = summarize_transfer_results(
        [
            {
                "checkpoint_name": "ckpt-a",
                "target_executor": "claude-sonnet",
                "transfer_executor": "gpt-4.1",
                "candidate_metrics": {"success_rate": 0.72, "mean_score": 0.7},
                "transfer_metrics": {"success_rate": 0.69, "mean_score": 0.66},
                "deltas": {"success_rate": 0.04, "mean_score": 0.05},
            }
        ]
    )

    assert ablations["by_kind"]["packet_field"][0]["target"] == "history"
    assert ablations["largest_drop"]["variant_id"] == "packet_field:history"
    assert transfer["executor_pairs"][0]["pair"] == "claude-sonnet->gpt-4.1"
    assert transfer["executor_pairs"][0]["transfer_success_rate"] == 0.69


def test_build_phase16_results_report_captures_failure_taxonomy_provenance_and_divergences(tmp_path):
    store = _seed_store(tmp_path)
    manifests = [
        BenchmarkRunManifest(
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
            score={"overall_score": 0.5, "focus_target_recall": 0.55},
        ),
        BenchmarkRunManifest(
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
            score={"overall_score": 0.75, "focus_target_recall": 0.8},
        ),
    ]
    transfer_payload = {
        "checkpoint_name": "ckpt-a",
        "target_executor": "claude-sonnet",
        "transfer_executor": "gpt-4.1",
        "candidate_metrics": {"success_rate": 0.72, "mean_score": 0.7},
        "transfer_metrics": {"success_rate": 0.69, "mean_score": 0.66},
        "deltas": {"success_rate": 0.04, "mean_score": 0.05},
    }
    report = build_phase16_results_report(
        store=store,
        benchmark_manifests=manifests,
        ablation_results=[{"variant_id": "packet_field:history", "kind": "packet_field", "target": "history", "overall_score_delta": -0.08}],
        transfer_results=[transfer_payload],
    )

    written = write_phase16_results_report(tmp_path / "report.json", report)
    reloaded = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert report["canonical_study"]["lift_summary"]["advisor_minus_baseline_overall_score"] == 0.25
    assert report["canonical_study"]["by_profile"]["coding-default"]["advisor"]["mean_overall_score"] == 0.75
    assert report["failure_taxonomy"]["categories"]["timeout_or_hang"]["count"] == 1
    assert report["provenance_coverage"]["reward_label_coverage"] == 1.0
    assert report["paper_divergences"][0]["status"] == "open"
    assert written == str(tmp_path / "report.json")
    assert reloaded["transfer_results"]["executor_pairs"][0]["pair"] == "claude-sonnet->gpt-4.1"


def test_supporting_helpers_surface_expected_defaults(tmp_path):
    store = _seed_store(tmp_path)

    taxonomy = build_failure_taxonomy(store)
    provenance = summarize_provenance_coverage(store)
    divergences = default_paper_divergences()

    assert taxonomy["total_failures"] == 1
    assert provenance["lineage_coverage"] == 1.0
    assert any(item["area"] == "training_recipe" for item in divergences)
