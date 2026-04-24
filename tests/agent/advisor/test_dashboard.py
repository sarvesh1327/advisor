from agent.advisor.core.schemas import AdviceBlock, AdvisorInputPacket, AdvisorOutcome, CandidateFile, RepoSummary
from agent.advisor.product.dashboard import build_advisor_activity_snapshot, render_advisor_activity_dashboard, simplify_run_title
from agent.advisor.storage.trace_store import AdvisorTraceStore


def _packet(run_id: str = "run-dash") -> AdvisorInputPacket:
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="answer the user's follow-up with context",
        task_type="conversation",
        repo={"path": "/tmp/advisor-repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["notes"], hotspots=["transcripts/chat-001.md"], file_tree_slice=["transcripts/chat-001.md"]),
        candidate_files=[CandidateFile(path="transcripts/chat-001.md", reason="latest conversation transcript", score=0.9)],
        recent_failures=[],
        constraints=["stay grounded in prior turns"],
        tool_limits={},
        acceptance_criteria=["answer is coherent"],
        token_budget=900,
    )


def test_build_advisor_activity_snapshot_includes_request_advice_and_outcome(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet()
    advice = AdviceBlock(task_type="conversation", recommended_plan=["read prior turns", "answer directly"], confidence=0.8)
    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-qwen25-3b-v1",
        advisor_profile_id="generalist",
        caller_id="hermes-main",
        latency_ms=42,
        prompt_hash="hash-1",
        injected_advice=advice,
        injected_rendered_advice="[Advisor hint]\n- read prior turns\n- answer directly",
    )
    store.record_outcome(
        AdvisorOutcome(
            run_id=packet.run_id,
            status="success",
            files_touched=[],
            retries=0,
            tests_run=[],
            summary="Produced a coherent answer.",
        )
    )

    snapshot = build_advisor_activity_snapshot(store, limit=5)

    assert snapshot["total_runs"] == 1
    assert snapshot["runs"][0]["advisor_profile_id"] == "generalist"
    assert snapshot["runs"][0]["advisor_used"] is True
    assert snapshot["runs"][0]["profile_badge"] == "Profile: generalist"
    assert snapshot["runs"][0]["caller_id"] == "hermes-main"
    assert snapshot["runs"][0]["caller_badge"] == "Caller: hermes-main"
    assert snapshot["runs"][0]["task_text"] == packet.task_text
    assert snapshot["runs"][0]["title"] == packet.task_text
    assert snapshot["runs"][0]["recommended_plan"] == ["read prior turns", "answer directly"]
    assert "Advisor hint" in snapshot["runs"][0]["injected_rendered_advice"]
    assert snapshot["runs"][0]["outcome"]["status"] == "success"


def test_simplify_run_title_strips_large_context_blocks_and_caps_length():
    noisy_task_text = """I see such big context on dashboard. can we turn titles of runs more simpler?

<memory-context>
very long recalled memory that should not appear in card title
</memory-context>

[Advisor middleware]
Run: run_123
Profile: researcher
Advice:
- review the brief
"""

    title = simplify_run_title(noisy_task_text)

    assert title == "I see such big context on dashboard. can we turn titles of runs more simpler?"
    assert "memory-context" not in title
    assert "Advisor middleware" not in title
    assert len(title) <= 90


def test_simplify_run_title_collapses_system_skill_and_maintenance_prompts():
    assert simplify_run_title('[SYSTEM: The user has invoked the "llm-wiki" skill, indicating they want you to follow its instructions.] --- name: llm-wiki') == "Skill: llm-wiki"
    assert simplify_run_title("Review the conversation above and consider saving or updating a skill if appropriate.\n\nFocus on: reusable workflow") == "Skill maintenance check"
    assert simplify_run_title("[SYSTEM: Background process proc_123 completed (exit code 0).]") == "Background process update"


def test_render_advisor_activity_dashboard_uses_simplified_title_not_raw_task_text():
    long_raw_text = """Fix dashboard titles

<memory-context>
this should stay out of h2
</memory-context>

[Advisor middleware]
Run: run_abc
"""

    html = render_advisor_activity_dashboard(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "total_runs": 1,
            "runs": [
                {
                    "run_id": "run-2",
                    "task_text": long_raw_text,
                    "title": simplify_run_title(long_raw_text),
                    "task_type": "conversation",
                    "advisor_profile_id": "generalist",
                    "advisor_used": True,
                    "profile_badge": "Profile: generalist",
                    "caller_id": "hermes-main",
                    "caller_badge": "Caller: hermes-main",
                    "recommended_plan": [],
                    "injected_rendered_advice": "",
                    "outcome": {},
                    "reward_label": None,
                }
            ],
        }
    )

    assert "<h2>Fix dashboard titles</h2>" in html
    assert "this should stay out of h2" not in html


def test_render_advisor_activity_dashboard_renders_run_cards():
    html = render_advisor_activity_dashboard(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "total_runs": 1,
            "runs": [
                {
                    "run_id": "run-1",
                    "started_at": "2026-04-23T00:00:00+00:00",
                    "task_text": "answer the follow-up",
                    "task_type": "conversation",
                    "advisor_profile_id": "generalist",
                    "advisor_used": True,
                    "profile_badge": "Profile: generalist",
                    "caller_id": "hermes-main",
                    "caller_badge": "Caller: hermes-main",
                    "recommended_plan": ["read prior turns", "answer directly"],
                    "injected_rendered_advice": "[Advisor hint] answer directly",
                    "outcome": {"status": "success", "summary": "done"},
                    "reward_label": None,
                }
            ],
        }
    )

    assert "Advisor activity dashboard" in html
    assert "Advisor used: yes" in html
    assert "Profile: generalist" in html
    assert "Caller: hermes-main" in html
    assert "answer the follow-up" in html
    assert "generalist" in html
    assert "answer directly" in html
