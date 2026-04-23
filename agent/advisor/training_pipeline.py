from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .settings import AdvisorSettings
from .trace_store import AdvisorTraceStore


@dataclass(frozen=True)
class RollbackPolicy:
    min_success_delta: float = -0.02
    min_score_delta: float = -0.03


@dataclass(frozen=True)
class AblationSpec:
    kind: Literal["packet_field", "advice_field", "reward_component"]
    target: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    student_model: str
    target_executor: str
    domain_mix: dict[str, float]
    training_mode: Literal["supervised", "preference"] = "supervised"
    preference_training_mode: Literal["dpo", "simpo", "orpo"] = "dpo"
    reward_preset: str = "balanced"
    transfer_executor: str | None = None
    ablations: list[AblationSpec] = field(default_factory=list)
    rollback: RollbackPolicy = field(default_factory=RollbackPolicy)


def build_dataset_manifest(
    store: AdvisorTraceStore,
    config: ExperimentConfig,
    *,
    min_quality_score: float = 0.0,
    advisor_profile_id: str | None = None,
) -> dict:
    runs = store.list_runs(include_context=True)
    examples = []
    for row in runs:
        reward = row.get("reward_label") or {}
        outcome = row.get("outcome") or {}
        if not reward or outcome.get("status") is None:
            continue
        resolved_profile_id = reward.get("advisor_profile_id") or row.get("advisor_profile_id")
        if advisor_profile_id and resolved_profile_id != advisor_profile_id:
            continue
        quality = float(reward.get("quality_score") or 0.0)
        example_type = reward.get("example_type") or "neutral"
        if quality < min_quality_score and example_type != "negative":
            continue
        examples.append(
            {
                "run_id": row["run_id"],
                "advisor_profile_id": resolved_profile_id,
                "split": reward.get("dataset_split") or "train",
                "example_type": example_type,
                "quality_score": quality,
                "hard_case_bucket": reward.get("hard_case_bucket"),
                "repo_path": row.get("repo_path"),
                "task_type": row.get("task_type"),
            }
        )
    settings = AdvisorSettings(reward_preset=config.reward_preset)
    counts = {
        "total_examples": len(examples),
        "positive_examples": sum(1 for item in examples if item["example_type"] == "positive"),
        "negative_examples": sum(1 for item in examples if item["example_type"] == "negative"),
        "neutral_examples": sum(1 for item in examples if item["example_type"] == "neutral"),
    }
    splits = sorted({item["split"] for item in examples})
    hard_case_buckets = sorted({item["hard_case_bucket"] for item in examples if item["hard_case_bucket"]})
    profile_ids = sorted({item["advisor_profile_id"] for item in examples if item.get("advisor_profile_id")})
    return {
        "experiment_id": config.experiment_id,
        "student_model": config.student_model,
        "target_executor": config.target_executor,
        "training_mode": config.training_mode,
        "preference_training_mode": config.preference_training_mode,
        "domain_mix": config.domain_mix,
        "reward_preset": config.reward_preset,
        "reward_weights": settings.reward_weights().__dict__,
        "advisor_profile_id": advisor_profile_id,
        "profile_ids": profile_ids,
        "profile_count": len(profile_ids),
        "counts": counts,
        "splits": splits,
        "hard_case_buckets": hard_case_buckets,
        "examples": examples,
    }


def evaluate_checkpoint(
    config: ExperimentConfig,
    *,
    checkpoint_name: str,
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    transfer_metrics: dict[str, float] | None = None,
) -> dict:
    deltas = {
        "success_rate": round(candidate_metrics.get("success_rate", 0.0) - baseline_metrics.get("success_rate", 0.0), 4),
        "mean_score": round(candidate_metrics.get("mean_score", 0.0) - baseline_metrics.get("mean_score", 0.0), 4),
    }
    return {
        "experiment_id": config.experiment_id,
        "checkpoint_name": checkpoint_name,
        "student_model": config.student_model,
        "target_executor": config.target_executor,
        "transfer_executor": config.transfer_executor,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "transfer_metrics": transfer_metrics,
        "deltas": deltas,
        "rollback": should_rollback(config.rollback, deltas),
    }



def generate_ablation_plans(config: ExperimentConfig) -> list[dict]:
    return [
        {
            "experiment_id": config.experiment_id,
            "kind": spec.kind,
            "target": spec.target,
            "variant_id": f"{spec.kind}:{spec.target}",
            "student_model": config.student_model,
            "target_executor": config.target_executor,
        }
        for spec in config.ablations
    ]



def should_rollback(policy: RollbackPolicy, deltas: dict[str, float]) -> bool:
    return (
        deltas.get("success_rate", 0.0) < policy.min_success_delta
        or deltas.get("mean_score", 0.0) < policy.min_score_delta
    )
