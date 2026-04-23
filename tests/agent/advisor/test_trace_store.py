from agent.advisor.core.schemas import (
    AdviceBlock,
    AdvisorArtifact,
    AdvisorCapabilityDescriptor,
    AdvisorContext,
    AdvisorHistoryEntry,
    AdvisorInputPacket,
    AdvisorOutcome,
    AdvisorTask,
    CandidateFile,
    ExecutorInjectionPolicy,
    RepoSummary,
)
from agent.advisor.rewards.reward_model import compute_reward_label
from agent.advisor.storage.trace_store import AdvisorTraceStore

DEFAULT_PROFILE_ID = "coding-default"


def _packet(run_id: str = "run-1"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="fix prompt builder",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=[],
        tool_limits={"write_allowed": True},
        acceptance_criteria=["tests pass"],
        token_budget=900,
    )


def test_trace_store_roundtrip(tmp_path):
    db_path = tmp_path / "advisor.db"
    store = AdvisorTraceStore(db_path)
    packet = _packet()
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=1, tests_run=["pytest -q"], review_verdict="pass")

    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-test",
        advisor_profile_id=DEFAULT_PROFILE_ID,
        latency_ms=10,
        prompt_hash="abc",
    )
    store.record_outcome(outcome)

    row = store.get_run("run-1")
    assert row is not None
    assert row["run_id"] == "run-1"
    assert row["advisor_profile_id"] == DEFAULT_PROFILE_ID
    assert row["advice"]["recommended_plan"] == ["inspect main.py"]
    assert row["injected_advice"]["recommended_plan"] == ["inspect main.py"]
    assert row["injection_policy"] == {
        "strategy": "prepend",
        "format": "plain_text",
        "min_confidence": 0.0,
        "include_confidence_note": True,
    }
    assert row["outcome"]["status"] == "success"
    assert row["input"]["task"]["domain"] == "coding"
    assert row["input"]["artifacts"][0]["locator"] == "main.py"
    assert row["input"]["context"]["metadata"]["repo"]["path"] == "/tmp/repo"


def test_trace_store_replays_canonical_generic_packet_state(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = AdvisorInputPacket(
        run_id="run-generic",
        task_text="update the hero layout from the latest screenshot",
        task_type="ui-update",
        repo={"path": "/tmp/ui", "branch": "main", "dirty": True},
        repo_summary=RepoSummary(
            modules=["ui"],
            hotspots=["ui/screens/home.png"],
            file_tree_slice=["ui/screens/home.png", "ui/layouts/home.json"],
        ),
        candidate_files=[CandidateFile(path="ui/screens/home.png", reason="changed screenshot", score=0.8)],
        recent_failures=[],
        constraints=["preserve spacing scale"],
        tool_limits={"image_read": True},
        acceptance_criteria=["hero matches updated mock"],
        token_budget=700,
        task=AdvisorTask(domain="image-ui", text="update the hero layout from the latest screenshot", type="ui-update"),
        context=AdvisorContext(
            summary="image-ui task context",
            metadata={"changed_files": ["ui/screens/home.png"], "packed": {"artifacts": 1, "history": 0}},
        ),
        artifacts=[
            AdvisorArtifact(
                kind="image",
                locator="ui/screens/home.png",
                description="latest screen capture",
                metadata={"changed": True, "region_hint": "hero"},
                score=1.2,
            )
        ],
        history=[
            AdvisorHistoryEntry(
                kind="prior-attempt",
                summary="previous pass drifted from the spacing grid",
                locator="ui/layouts/home.json",
                metadata={"region_hint": "hero"},
            )
        ],
        domain_capabilities=[
            AdvisorCapabilityDescriptor(
                domain="image-ui",
                supported_artifact_kinds=["image", "layout"],
                supported_packet_fields=["task", "context", "artifacts", "constraints", "history", "acceptance_criteria"],
                supports_changed_artifacts=True,
                supports_symbol_regions=True,
            )
        ],
    )

    store.record_task_run(
        packet,
        AdviceBlock(
            task_type="ui-update",
            recommended_plan=["inspect ui/screens/home.png"],
            confidence=0.7,
            injection_policy=ExecutorInjectionPolicy(min_confidence=0.5),
        ),
        advisor_model="advisor-test",
        advisor_profile_id=DEFAULT_PROFILE_ID,
        latency_ms=12,
        prompt_hash="def",
        injected_rendered_advice="[Advisor hint — use as guidance, not authority]",
    )

    replay = store.get_run("run-generic")

    assert replay is not None
    assert replay["advisor_profile_id"] == DEFAULT_PROFILE_ID
    assert replay["input"]["task"]["domain"] == "image-ui"
    assert replay["input"]["context"]["summary"] == "image-ui task context"
    assert replay["input"]["artifacts"] == [
        {
            "kind": "image",
            "locator": "ui/screens/home.png",
            "description": "latest screen capture",
            "metadata": {"changed": True, "region_hint": "hero"},
            "score": 1.2,
        }
    ]
    assert replay["input"]["history"][0]["kind"] == "prior-attempt"
    assert replay["input"]["history"][0]["metadata"]["region_hint"] == "hero"
    assert replay["input"]["domain_capabilities"][0]["supports_symbol_regions"] is True
    assert replay["injection_policy"]["min_confidence"] == 0.5
    assert replay["injected_rendered_advice"] == "[Advisor hint — use as guidance, not authority]"


def test_trace_store_records_reward_labels(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet("run-reward")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(
        run_id="run-reward",
        status="success",
        files_touched=["main.py"],
        retries=1,
        tests_run=["pytest -q"],
        review_verdict="pass",
    )

    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-test",
        advisor_profile_id=DEFAULT_PROFILE_ID,
        latency_ms=10,
        prompt_hash="abc",
    )
    store.record_outcome(outcome)
    reward_label = compute_reward_label(packet, advice, outcome, human_rating=5.0)

    store.record_reward_label(reward_label)

    row = store.get_run("run-reward")
    assert row is not None
    assert row["reward_label"]["run_id"] == "run-reward"
    assert row["reward_label"]["components"]["task_success"] == 1.0
    assert row["reward_label"]["example_type"] == "positive"


def test_trace_store_roundtrips_profile_aware_reward_metadata(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet("run-profile-reward")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(
        run_id="run-profile-reward",
        status="success",
        files_touched=["main.py"],
        retries=1,
        tests_run=["pytest -q"],
        review_verdict="pass",
    )

    store.record_task_run(
        packet,
        advice,
        advisor_model="advisor-test",
        advisor_profile_id=DEFAULT_PROFILE_ID,
        latency_ms=10,
        prompt_hash="profile-reward",
    )
    store.record_outcome(outcome)
    store.record_reward_label(
        {
            "run_id": "run-profile-reward",
            "advisor_profile_id": DEFAULT_PROFILE_ID,
            "reward_profile_id": "coding_swe_efficiency",
            "reward_formula": "coding_swe_efficiency",
            "reward_version": "coding-swe-efficiency-v1",
            "raw_reward": 0.875,
            "total_reward": 0.875,
            "quality_score": 0.875,
            "reward_diagnostics": {"steps": 10, "max_steps": 40},
            "dataset_split": "train",
            "example_type": "positive",
            "hard_case_bucket": None,
            "notes": ["pass"],
        }
    )

    row = store.get_run("run-profile-reward")
    assert row is not None
    assert row["reward_label"]["advisor_profile_id"] == DEFAULT_PROFILE_ID
    assert row["reward_label"]["reward_profile_id"] == "coding_swe_efficiency"
    assert row["reward_label"]["reward_formula"] == "coding_swe_efficiency"
    assert row["reward_label"]["reward_diagnostics"]["steps"] == 10


def test_trace_store_prefers_prior_failures_with_changed_file_overlap(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    repo_path = "/tmp/ui"
    for run_id, task_text, files_touched, summary in (
        ("run-match", "revise product copy", ["ui/mockups/home.png"], "copy change broke the hero alignment"),
        ("run-token", "refresh hero image layout", ["docs/brief.md"], "hero image still misaligned"),
    ):
        store.record_task_run(
            AdvisorInputPacket(
                run_id=run_id,
                task_text=task_text,
                task_type="bugfix",
                repo={"path": repo_path, "branch": "main", "dirty": False},
                repo_summary=RepoSummary(modules=["ui"], hotspots=files_touched, file_tree_slice=files_touched),
                candidate_files=[CandidateFile(path=files_touched[0], reason="match", score=0.7)],
                recent_failures=[],
                constraints=[],
                tool_limits={},
                acceptance_criteria=[],
                token_budget=600,
            ),
            AdviceBlock(task_type="bugfix", recommended_plan=["inspect file"], confidence=0.5),
            advisor_model="advisor-test",
            advisor_profile_id=DEFAULT_PROFILE_ID,
            latency_ms=5,
            prompt_hash=f"hash-{run_id}",
        )
        store.record_outcome(
            AdvisorOutcome(run_id=run_id, status="failure", files_touched=files_touched, summary=summary)
        )

    failures = store.find_recent_failures(
        "refresh the hero image",
        repo_path,
        limit=2,
        changed_files=["ui/mockups/home.png"],
    )

    assert [failure.summary for failure in failures] == [
        "copy change broke the hero alignment",
        "hero image still misaligned",
    ]
