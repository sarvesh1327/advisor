from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from .eval_fixtures import EvalFixture
from .replay import evaluate_replay_run
from .trace_store import AdvisorTraceStore


class BenchmarkCase(BaseModel):
    fixture_id: str
    domain: str
    split: Literal["train_pool", "validation", "test"]
    description: str


class BenchmarkSuite(BaseModel):
    suite_id: str
    cases: list[BenchmarkCase] = Field(default_factory=list)


class BenchmarkRunManifest(BaseModel):
    run_id: str
    fixture_id: str
    domain: str
    split: Literal["train_pool", "validation", "test"]
    packet_hash: str
    executor_config: dict = Field(default_factory=dict)
    verifier_set: list[str] = Field(default_factory=list)
    routing_arm: Literal["baseline", "advisor"]
    reward_version: str
    score: dict = Field(default_factory=dict)


def freeze_benchmark_suite(suite_id: str, fixtures: list[EvalFixture]) -> BenchmarkSuite:
    cases = []
    for fixture in sorted(fixtures, key=lambda item: (item.domain, item.fixture_id)):
        digest = hashlib.sha256(f"{suite_id}|{fixture.domain}|{fixture.fixture_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket <= 2 else "train_pool"
        cases.append(
            BenchmarkCase(
                fixture_id=fixture.fixture_id,
                domain=fixture.domain,
                split=split,
                description=fixture.description,
            )
        )
    return BenchmarkSuite(suite_id=suite_id, cases=cases)


def build_benchmark_run_manifest(
    *,
    store: AdvisorTraceStore,
    run_id: str,
    fixture: EvalFixture,
    split: Literal["train_pool", "validation", "test"],
) -> BenchmarkRunManifest:
    score_payload = evaluate_replay_run(store, run_id, fixture)
    lineage = store.get_lineage(run_id)
    if lineage is None:
        raise ValueError(f"missing lineage for benchmark run: {run_id}")
    manifest = lineage.get("manifest") or {}
    replay_inputs = manifest.get("replay_inputs") or {}
    executor = manifest.get("executor") or {}
    verifiers = manifest.get("verifiers") or []
    routing = manifest.get("routing_decision") or {}
    run_row = store.get_run(run_id) or {}
    reward_label = run_row.get("reward_label") or {}
    return BenchmarkRunManifest(
        run_id=run_id,
        fixture_id=fixture.fixture_id,
        domain=fixture.domain,
        split=split,
        packet_hash=replay_inputs.get("packet_hash") or _hash_packet(score_payload.get("input") or {}),
        executor_config=executor,
        verifier_set=sorted(item.get("name") for item in verifiers if item.get("name")),
        routing_arm=routing.get("arm", "baseline"),
        reward_version=reward_label.get("reward_version", "unknown"),
        score=score_payload.get("score") or {},
    )


def compare_benchmark_arms(manifests: list[BenchmarkRunManifest]) -> dict:
    manifests = sorted(
        manifests,
        key=lambda item: (item.split, item.domain, item.fixture_id, item.routing_arm, item.run_id),
    )
    arm_summary: dict[str, dict] = {}
    by_split: dict[str, dict[str, dict]] = {}
    by_domain: dict[str, dict[str, dict]] = {}
    ablation_axes = {
        "domains": sorted({item.domain for item in manifests}),
        "executor_kinds": sorted({item.executor_config.get("kind", "unknown") for item in manifests}),
        "reward_versions": sorted({item.reward_version for item in manifests}),
        "splits": sorted({item.split for item in manifests}),
        "verifier_sets": sorted({"|".join(item.verifier_set) if item.verifier_set else "" for item in manifests}),
    }
    for item in manifests:
        overall = float(item.score.get("overall_score", 0.0))
        recall = float(item.score.get("focus_target_recall", 0.0))
        arm_bucket = arm_summary.setdefault(item.routing_arm, {"count": 0, "overall_total": 0.0, "recall_total": 0.0})
        arm_bucket["count"] += 1
        arm_bucket["overall_total"] += overall
        arm_bucket["recall_total"] += recall

        split_bucket = by_split.setdefault(item.split, {}).setdefault(item.routing_arm, {"count": 0, "overall_total": 0.0})
        split_bucket["count"] += 1
        split_bucket["overall_total"] += overall

        domain_bucket = by_domain.setdefault(item.domain, {}).setdefault(item.routing_arm, {"count": 0, "overall_total": 0.0})
        domain_bucket["count"] += 1
        domain_bucket["overall_total"] += overall

    finalized_arm_summary = {
        arm: {
            "count": bucket["count"],
            "mean_overall_score": round(bucket["overall_total"] / bucket["count"], 4) if bucket["count"] else 0.0,
            "mean_focus_target_recall": round(bucket["recall_total"] / bucket["count"], 4) if bucket["count"] else 0.0,
        }
        for arm, bucket in arm_summary.items()
    }
    finalized_by_split = {
        split: {
            arm: {
                "count": bucket["count"],
                "mean_overall_score": round(bucket["overall_total"] / bucket["count"], 4) if bucket["count"] else 0.0,
            }
            for arm, bucket in arms.items()
        }
        for split, arms in by_split.items()
    }
    finalized_by_domain = {
        domain: {
            arm: {
                "count": bucket["count"],
                "mean_overall_score": round(bucket["overall_total"] / bucket["count"], 4) if bucket["count"] else 0.0,
            }
            for arm, bucket in arms.items()
        }
        for domain, arms in by_domain.items()
    }

    advisor_mean = finalized_arm_summary.get("advisor", {}).get("mean_overall_score", 0.0)
    baseline_mean = finalized_arm_summary.get("baseline", {}).get("mean_overall_score", 0.0)
    return {
        "run_count": len(manifests),
        "arm_summary": finalized_arm_summary,
        "by_split": finalized_by_split,
        "by_domain": finalized_by_domain,
        "deltas": {
            "advisor_minus_baseline": {
                "overall_score": round(advisor_mean - baseline_mean, 4),
            }
        },
        "ablation_axes": ablation_axes,
    }


def _hash_packet(packet: dict) -> str:
    return hashlib.sha256(json.dumps(packet, sort_keys=True).encode("utf-8")).hexdigest()
