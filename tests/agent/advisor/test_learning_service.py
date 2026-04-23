from pathlib import Path

from agent.advisor.core.settings import AdvisorSettings
from agent.advisor.learning.controller import AutonomousLearningController
from agent.advisor.learning.service import run_autonomous_learning_service
from agent.advisor.storage.trace_store import AdvisorTraceStore
from tests.agent.advisor.test_learning_controller import _seed_dogfood_run

ADVISOR_REPO = str(Path("/Users/clawuser/Desktop/Black-box steer/Advisor").expanduser())


def test_run_autonomous_learning_service_ticks_without_duplicate_cycles(tmp_path, monkeypatch):
    settings, store, _ = _seed_dogfood_run(tmp_path, "run-service-1")
    _seed_dogfood_run(tmp_path, "run-service-2")

    calls = {"count": 0}

    def fake_cycle_runner(queue, **kwargs):
        del queue, kwargs
        calls["count"] += 1
        return {
            "train_job": {"job_id": f"job-train-{calls['count']}", "result": {"checkpoint_id": "ckpt-1"}},
            "eval_job": {"job_id": f"job-eval-{calls['count']}", "result": {"promote": False}},
            "promote_job": None,
            "promoted": False,
        }

    monkeypatch.setattr("agent.advisor.learning.controller.run_continuous_training_cycle", fake_cycle_runner)
    controller = AutonomousLearningController(settings=settings, trace_store=store)

    result = run_autonomous_learning_service(settings=settings, controller=controller, max_ticks=2, sleep_fn=lambda _: None)

    assert result["tick_count"] == 2
    assert calls["count"] == 1
    assert result["controller_status"]["profiles"]["coding-default"]["consumed_run_ids"]


def test_run_autonomous_learning_service_reports_controller_pause(tmp_path):
    settings = AdvisorSettings(
        enabled=True,
        trace_db_path=str(tmp_path / "advisor.db"),
        event_log_path=str(tmp_path / "events.jsonl"),
        advisor_profiles_path=str(Path(ADVISOR_REPO) / "config" / "advisor_profiles.toml"),
    )
    store = AdvisorTraceStore(settings.trace_db_path)
    controller = AutonomousLearningController(settings=settings, trace_store=store)
    controller.pause_controller("maintenance")

    result = run_autonomous_learning_service(settings=settings, controller=controller, max_ticks=1, sleep_fn=lambda _: None)

    assert result["last_result"]["controller_paused"] is True
    assert result["controller_status"]["controller_paused_reason"] == "maintenance"
