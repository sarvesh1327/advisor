from __future__ import annotations

import json
from pathlib import Path

from agent.advisor.evaluation.benchmark import BenchmarkRunManifest, compare_benchmark_arms
from agent.advisor.storage.trace_store import AdvisorTraceStore


def summarize_canonical_study(benchmark_manifests: list[BenchmarkRunManifest]) -> dict:
    comparison = compare_benchmark_arms(benchmark_manifests)
    arm_summary = comparison.get("arm_summary") or {}
    baseline = arm_summary.get("baseline", {})
    advisor = arm_summary.get("advisor", {})
    return {
        "protocol": {
            "parity_rule": "same executor, same verifier set, advice injection only",
            "frozen_suite_required": True,
            "run_count": comparison.get("run_count", 0),
        },
        "arm_summary": arm_summary,
        "by_split": comparison.get("by_split", {}),
        "by_domain": comparison.get("by_domain", {}),
        "by_profile": comparison.get("by_profile", {}),
        "ablation_axes": comparison.get("ablation_axes", {}),
        "lift_summary": {
            "baseline_mean_overall_score": baseline.get("mean_overall_score", 0.0),
            "advisor_mean_overall_score": advisor.get("mean_overall_score", 0.0),
            "advisor_minus_baseline_overall_score": comparison.get("deltas", {})
            .get("advisor_minus_baseline", {})
            .get("overall_score", 0.0),
        },
    }


def summarize_ablation_results(ablation_results: list[dict]) -> dict:
    ordered = sorted(
        [dict(item) for item in ablation_results],
        key=lambda item: (item.get("kind", ""), item.get("target", ""), item.get("variant_id", "")),
    )
    by_kind: dict[str, list[dict]] = {}
    largest_drop = None
    largest_gain = None
    for item in ordered:
        delta = float(item.get("overall_score_delta", 0.0))
        normalized = {**item, "overall_score_delta": round(delta, 4)}
        by_kind.setdefault(item.get("kind", "unknown"), []).append(normalized)
        if largest_drop is None or delta < float(largest_drop.get("overall_score_delta", 0.0)):
            largest_drop = normalized
        if largest_gain is None or delta > float(largest_gain.get("overall_score_delta", 0.0)):
            largest_gain = normalized
    return {
        "variant_count": len(ordered),
        "by_kind": by_kind,
        "largest_drop": largest_drop,
        "largest_gain": largest_gain,
    }


def summarize_transfer_results(transfer_results: list[dict]) -> dict:
    executor_pairs = []
    for item in sorted(transfer_results, key=lambda row: (row.get("target_executor", ""), row.get("transfer_executor", ""), row.get("checkpoint_name", ""))):
        candidate = item.get("candidate_metrics") or {}
        transfer = item.get("transfer_metrics") or {}
        executor_pairs.append(
            {
                "checkpoint_name": item.get("checkpoint_name"),
                "pair": f"{item.get('target_executor')}->{item.get('transfer_executor')}",
                "target_success_rate": round(float(candidate.get("success_rate", 0.0)), 4),
                "transfer_success_rate": round(float(transfer.get("success_rate", 0.0)), 4),
                "target_mean_score": round(float(candidate.get("mean_score", 0.0)), 4),
                "transfer_mean_score": round(float(transfer.get("mean_score", 0.0)), 4),
                "delta_success_rate": round(float((item.get("deltas") or {}).get("success_rate", 0.0)), 4),
                "delta_mean_score": round(float((item.get("deltas") or {}).get("mean_score", 0.0)), 4),
            }
        )
    return {
        "pair_count": len(executor_pairs),
        "executor_pairs": executor_pairs,
    }


def build_failure_taxonomy(store: AdvisorTraceStore) -> dict:
    categories = {
        "timeout_or_hang": {"count": 0, "run_ids": []},
        "test_regression": {"count": 0, "run_ids": []},
        "runtime_or_exception": {"count": 0, "run_ids": []},
        "unknown": {"count": 0, "run_ids": []},
    }
    failures = 0
    for row in store.list_runs(include_context=False):
        outcome = row.get("outcome") or {}
        if outcome.get("status") != "failure":
            continue
        failures += 1
        bucket = _categorize_failure(outcome)
        categories[bucket]["count"] += 1
        categories[bucket]["run_ids"].append(row["run_id"])
    return {
        "total_failures": failures,
        "categories": categories,
    }


def summarize_provenance_coverage(store: AdvisorTraceStore) -> dict:
    runs = store.list_runs(include_context=False)
    total_runs = len(runs)
    lineage_count = 0
    reward_count = 0
    both_count = 0
    for row in runs:
        has_lineage = store.get_lineage(row["run_id"]) is not None
        has_reward = bool(row.get("reward_label"))
        lineage_count += 1 if has_lineage else 0
        reward_count += 1 if has_reward else 0
        both_count += 1 if has_lineage and has_reward else 0
    return {
        "total_runs": total_runs,
        "lineage_runs": lineage_count,
        "reward_labeled_runs": reward_count,
        "lineage_coverage": _ratio(lineage_count, total_runs),
        "reward_label_coverage": _ratio(reward_count, total_runs),
        "joint_provenance_coverage": _ratio(both_count, total_runs),
    }


def default_paper_divergences() -> list[dict]:
    return [
        {
            "area": "training_recipe",
            "status": "open",
            "detail": "Repo uses a production-oriented training/checkpoint scaffold rather than the paper's exact student-teacher optimization recipe.",
        },
        {
            "area": "benchmark_corpus",
            "status": "open",
            "detail": "Frozen suites are repo-native benchmark manifests, not the paper's original benchmark corpus and annotation pipeline.",
        },
        {
            "area": "reward_labeling",
            "status": "open",
            "detail": "Reward labels are derived from local verifier/live-run signals and human usefulness fields rather than the paper's full labeling stack.",
        },
        {
            "area": "executor_surface",
            "status": "intentional_productization",
            "detail": "Executor/verifier/operator surfaces were productized to support real deployments while preserving the evaluation contract.",
        },
    ]


def build_phase16_results_report(
    *,
    store: AdvisorTraceStore,
    benchmark_manifests: list[BenchmarkRunManifest],
    ablation_results: list[dict] | None = None,
    transfer_results: list[dict] | None = None,
    paper_divergences: list[dict] | None = None,
) -> dict:
    return {
        "canonical_study": summarize_canonical_study(benchmark_manifests),
        "ablation_results": summarize_ablation_results(ablation_results or []),
        "transfer_results": summarize_transfer_results(transfer_results or []),
        "failure_taxonomy": build_failure_taxonomy(store),
        "provenance_coverage": summarize_provenance_coverage(store),
        "paper_divergences": paper_divergences or default_paper_divergences(),
    }


def write_phase16_results_report(path: str | Path, report: dict) -> str:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def _categorize_failure(outcome: dict) -> str:
    text = " ".join(
        part for part in [outcome.get("summary"), outcome.get("review_verdict"), " ".join(outcome.get("tests_run") or [])] if part
    ).lower()
    if any(token in text for token in ("timeout", "timed out", "hang", "stalled")):
        return "timeout_or_hang"
    if any(token in text for token in ("pytest", "test", "assert", "regress")):
        return "test_regression"
    if any(token in text for token in ("exception", "traceback", "error", "crash")):
        return "runtime_or_exception"
    return "unknown"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)
