from __future__ import annotations

import time
from typing import Callable

from agent.advisor.core.settings import AdvisorSettings

from .controller import AutonomousLearningController


def run_autonomous_learning_service(
    *,
    settings: AdvisorSettings,
    controller: AutonomousLearningController | None = None,
    max_ticks: int | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    active_controller = controller or AutonomousLearningController(settings=settings)
    tick_count = 0
    last_result: dict = {}
    while max_ticks is None or tick_count < max_ticks:
        last_result = active_controller.tick()
        tick_count += 1
        if max_ticks is not None and tick_count >= max_ticks:
            break
        sleep_fn(float(active_controller.load_state().policy.tick_interval_seconds))
    return {
        "tick_count": tick_count,
        "last_result": last_result,
        "controller_status": active_controller.controller_status(),
    }
