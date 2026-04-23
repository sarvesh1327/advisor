from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, CandidateFile, RepoSummary
from agent.advisor.domain_rewards.coding import compute_coding_exact_answer_reward, compute_coding_swe_efficiency_reward
from agent.advisor.domain_rewards.research import compute_research_writing_match_reward
from agent.advisor.domain_rewards.ui import compute_ui_edit_from_screenshot_reward, compute_ui_from_text_layout_reward
from agent.advisor.profiles import AdvisorProfile
from agent.advisor.rewards.reward_registry import RewardRegistry
from agent.advisor.rewards.reward_specs import RewardSpec


def _packet(run_id: str = "run-domain"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="repair the execution loop",
        task_type="bugfix",
        repo={"path": "/tmp/advisor-repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["agent"], hotspots=["gateway.py"], file_tree_slice=["gateway.py"]),
        candidate_files=[CandidateFile(path="gateway.py", reason="entrypoint", score=0.9)],
        recent_failures=[],
        constraints=["tests pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["all verifiers pass"],
        token_budget=900,
    )


def test_reward_registry_resolves_profile_reward_spec_and_computes_scalar_reward():
    registry = RewardRegistry(
        specs={
            "coding_swe_efficiency": RewardSpec(
                reward_spec_id="coding_swe_efficiency",
                domain="coding",
                formula_name="coding_swe_efficiency",
                reward_version="coding-swe-efficiency-v1",
            )
        }
    )
    profile = AdvisorProfile(
        profile_id="coding-default",
        domain="coding",
        description="Default coding profile",
        reward_spec_id="coding_swe_efficiency",
    )
    packet = _packet()
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id=packet.run_id, status="success", files_touched=["gateway.py"], retries=1, tests_run=["pytest -q"])

    label = registry.compute_reward_for_profile(
        profile,
        packet,
        advice,
        outcome,
        executor_result={"status": "success", "metadata": {"steps": 10}},
        verifier_results=[{"status": "pass", "metadata": {}}],
    )

    assert label.advisor_profile_id == "coding-default"
    assert label.reward_profile_id == "coding_swe_efficiency"
    assert label.reward_formula == "coding_swe_efficiency"
    assert label.reward_version == "coding-swe-efficiency-v1"
    assert label.raw_reward == 0.875
    assert label.total_reward == 0.875
    assert label.components is None
    assert label.reward_diagnostics["steps"] == 10



def test_reward_registry_rejects_unknown_reward_spec_for_profile():
    registry = RewardRegistry(specs={})
    profile = AdvisorProfile(profile_id="custom", domain="coding", reward_spec_id="missing")

    try:
        registry.resolve_spec_for_profile(profile)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing reward spec to fail")



def test_coding_reward_functions_cover_efficiency_and_exact_answer_cases():
    swe_reward, swe_diagnostics = compute_coding_swe_efficiency_reward(resolved=True, steps=10)
    exact_reward, exact_diagnostics = compute_coding_exact_answer_reward(exact_correct=False)

    assert swe_reward == 0.875
    assert swe_diagnostics == {"resolved": True, "steps": 10, "max_steps": 40}
    assert exact_reward == 0.0
    assert exact_diagnostics == {"exact_correct": False}



def test_ui_reward_functions_cover_layout_and_screenshot_paths():
    layout_reward, layout_diagnostics = compute_ui_from_text_layout_reward(
        render_valid=True,
        hard_constraint_pass_rate=0.8,
        soft_style_score=0.6,
    )
    screenshot_reward, screenshot_diagnostics = compute_ui_edit_from_screenshot_reward(
        render_valid=True,
        screenshot_similarity=0.9,
        constraint_pass_rate=0.5,
    )

    assert layout_reward == 0.75
    assert layout_diagnostics["hard_constraint_pass_rate"] == 0.8
    assert screenshot_reward == 0.78
    assert screenshot_diagnostics["screenshot_similarity"] == 0.9



def test_reward_registry_preserves_explicit_zero_constraint_metric_over_fallback_value():
    registry = RewardRegistry.default()
    packet = _packet("run-zero-metric")
    advice = AdviceBlock(task_type="ui-update", recommended_plan=["inspect gateway.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id=packet.run_id, status="success", files_touched=["gateway.py"], retries=0, tests_run=[])

    label = registry.compute_reward_for_profile(
        AdvisorProfile(
            profile_id="image-ui-screenshot",
            domain="image-ui",
            reward_spec_id="ui_edit_from_screenshot",
        ),
        packet,
        advice,
        outcome,
        executor_result={"status": "success", "metadata": {"render_valid": True}},
        verifier_results=[
            {
                "status": "pass",
                "metadata": {
                    "screenshot_similarity": 1.0,
                    "constraint_pass_rate": 0.0,
                    "hard_constraint_pass_rate": 1.0,
                },
            }
        ],
    )

    assert label.raw_reward == 0.7
    assert label.reward_diagnostics["constraint_pass_rate"] == 0.0



def test_reward_registry_resolves_text_ui_profile_to_ui_layout_reward():
    registry = RewardRegistry.default()
    packet = _packet("run-text-ui")
    packet.task.domain = "text-ui"
    advice = AdviceBlock(task_type="ui-update", recommended_plan=["draft the layout spec"], confidence=0.8)
    outcome = AdvisorOutcome(run_id=packet.run_id, status="success", files_touched=["gateway.py"], retries=0, tests_run=[])

    label = registry.compute_for_profile_id(
        "text-ui",
        packet,
        advice,
        outcome,
        executor_result={"status": "success", "metadata": {"render_valid": True}},
        verifier_results=[{"status": "pass", "metadata": {"hard_constraint_pass_rate": 0.8, "soft_style_score": 0.6}}],
    )

    assert label.advisor_profile_id == "text-ui"
    assert label.reward_profile_id == "ui_from_text_layout"
    assert label.reward_formula == "ui_from_text_layout"
    assert label.raw_reward == 0.75



def test_reward_registry_handles_non_numeric_steps_by_falling_back_to_retry_count():
    registry = RewardRegistry.default()
    packet = _packet("run-bad-steps")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect gateway.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id=packet.run_id, status="success", files_touched=["gateway.py"], retries=2, tests_run=[])

    label = registry.compute_for_profile_id(
        "coding-default",
        packet,
        advice,
        outcome,
        executor_result={"status": "success", "metadata": {"steps": "bad-value"}},
        verifier_results=[{"status": "pass", "metadata": {}}],
    )

    assert label.raw_reward == 0.9625
    assert label.reward_diagnostics["steps"] == 3



def test_research_reward_function_averages_grounding_constraint_and_coverage_scores():
    reward, diagnostics = compute_research_writing_match_reward(
        grounding_score=0.9,
        constraint_compliance=0.6,
        coverage_score=0.75,
    )

    assert reward == 0.75
    assert diagnostics == {
        "grounding_score": 0.9,
        "constraint_compliance": 0.6,
        "coverage_score": 0.75,
    }
