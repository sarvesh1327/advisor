import json

from agent.advisor.hardening import (
    BenchmarkReleasePolicy,
    build_alert_summary,
    build_deployment_hardening_profile,
    evaluate_release_gate,
    export_product_bundle,
    import_product_bundle,
    lock_truth_surface_contract,
)
from agent.advisor.settings import AdvisorSettings


def _results_report(*, lift: float = 0.12, reward_coverage: float = 1.0, lineage_coverage: float = 1.0, open_divergences: int = 1):
    divergences = [
        {"area": f"area-{idx}", "status": "open", "detail": "still open"}
        for idx in range(open_divergences)
    ]
    return {
        "canonical_study": {
            "lift_summary": {
                "advisor_minus_baseline_overall_score": lift,
            }
        },
        "provenance_coverage": {
            "reward_label_coverage": reward_coverage,
            "lineage_coverage": lineage_coverage,
        },
        "paper_divergences": divergences,
    }


def test_release_gate_enforces_regression_thresholds_and_alerts():
    passing = evaluate_release_gate(_results_report(), BenchmarkReleasePolicy())
    failing = evaluate_release_gate(
        _results_report(lift=-0.02, reward_coverage=0.6, lineage_coverage=0.7, open_divergences=3),
        BenchmarkReleasePolicy(min_overall_lift=0.0, min_reward_label_coverage=0.9, min_lineage_coverage=0.9, max_open_divergences=1),
    )
    alerts = build_alert_summary(failing)

    assert passing["pass"] is True
    assert failing["pass"] is False
    assert failing["checks"]["overall_lift"]["pass"] is False
    assert failing["checks"]["reward_label_coverage"]["pass"] is False
    assert alerts["severity"] == "critical"
    assert "release gate failed" in alerts["summary"].lower()


def test_build_deployment_hardening_profile_sets_hosted_auth_and_isolation_requirements(tmp_path):
    single_tenant = build_deployment_hardening_profile(mode="single_tenant", state_root=tmp_path / "single")
    hosted = build_deployment_hardening_profile(mode="hosted", state_root=tmp_path / "hosted")

    assert single_tenant.auth["mode"] == "local-boundary"
    assert hosted.auth["mode"] == "external-auth-proxy"
    assert hosted.tenancy["tenant_id_required"] is True
    assert hosted.isolation["storage_strategy"] == "per-tenant-root"
    assert len(hosted.operator_runbooks) >= 3


def test_export_and_import_product_bundle_preserve_state_and_contract_manifest(tmp_path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    trace_db = state_root / "advisor.db"
    event_log = state_root / "events.jsonl"
    archive_dir = state_root / "archive"
    archive_dir.mkdir()
    (archive_dir / "runs.jsonl").write_text('{"run_id":"run-1"}\n', encoding="utf-8")
    trace_db.write_text("sqlite-placeholder", encoding="utf-8")
    event_log.write_text('{"event":"ok"}\n', encoding="utf-8")
    settings = AdvisorSettings(trace_db_path=str(trace_db), event_log_path=str(event_log))

    bundle_path = export_product_bundle(output_dir=tmp_path / "bundle", settings=settings)
    imported_root = import_product_bundle(bundle_path=bundle_path, target_root=tmp_path / "restored")
    contract = json.loads((tmp_path / "bundle" / "truth-surface-contract.json").read_text(encoding="utf-8"))

    assert (tmp_path / "bundle" / "state" / "advisor.db").exists()
    assert (imported_root / "state" / "advisor.db").read_text(encoding="utf-8") == "sqlite-placeholder"
    assert contract["benchmark_contract"]["version"] == "v1"
    assert contract["reward_contract"]["version"] == "v1"


def test_lock_truth_surface_contract_writes_versioned_manifest(tmp_path):
    contract_path = lock_truth_surface_contract(tmp_path / "contract.json")
    contract = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))

    assert contract_path == str(tmp_path / "contract.json")
    assert contract["packet_schema"]["version"] == "v1"
    assert contract["advice_schema"]["version"] == "v1"
    assert contract["experiment_report_contract"]["version"] == "v1"
