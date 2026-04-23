import json

from agent.advisor.labeling import export_training_examples
from agent.advisor.reward_model import compute_reward_label
from agent.advisor.schemas import (
    AdviceBlock,
    AdvisorInputPacket,
    AdvisorOutcome,
    CandidateFile,
    RepoSummary,
)
from agent.advisor.trace_store import AdvisorTraceStore


def _packet(run_id: str):
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


def test_export_training_examples_writes_jsonl(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    packet = _packet("run-1")
    advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.8)
    outcome = AdvisorOutcome(run_id="run-1", status="success", files_touched=["main.py"], retries=0, tests_run=["pytest -q"], review_verdict="pass")
    store.record_task_run(packet, advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="abc")
    store.record_outcome(outcome)
    store.record_reward_label(compute_reward_label(packet, advice, outcome, human_rating=5.0))

    out = tmp_path / "train.jsonl"
    count = export_training_examples(store, out)
    assert count == 1
    rows = out.read_text().strip().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["run_id"] == "run-1"
    assert payload["input"]["task_text"] == "fix prompt builder"
    assert payload["target_advice"]["recommended_plan"] == ["inspect main.py"]
    assert payload["reward_label"]["example_type"] == "positive"
    assert payload["quality_score"] == payload["reward_label"]["quality_score"]
    assert payload["reward_label"]["reward_profile_id"] == "legacy-generic"


def test_export_training_examples_filters_low_quality_and_keeps_negative_examples(tmp_path):
    store = AdvisorTraceStore(tmp_path / "advisor.db")

    positive_packet = _packet("run-positive")
    positive_advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect main.py"], confidence=0.9)
    positive_outcome = AdvisorOutcome(
        run_id="run-positive",
        status="success",
        files_touched=["main.py"],
        retries=0,
        tests_run=["pytest -q"],
        review_verdict="pass",
    )
    store.record_task_run(positive_packet, positive_advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="pos")
    store.record_outcome(positive_outcome)
    store.record_reward_label(compute_reward_label(positive_packet, positive_advice, positive_outcome, human_rating=5.0))

    negative_packet = _packet("run-negative")
    negative_advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect docs.md"], confidence=0.3)
    negative_outcome = AdvisorOutcome(
        run_id="run-negative",
        status="failure",
        files_touched=["main.py"],
        retries=4,
        tests_run=[],
        review_verdict="constraint drift",
    )
    store.record_task_run(negative_packet, negative_advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="neg")
    store.record_outcome(negative_outcome)
    store.record_reward_label(
        compute_reward_label(
            negative_packet,
            negative_advice,
            negative_outcome,
            constraint_violations=["constraint drift"],
        )
    )

    low_quality_packet = _packet("run-low")
    low_quality_packet.task_text = "fix unrelated docs"
    low_quality_advice = AdviceBlock(task_type="bugfix", recommended_plan=["inspect unrelated.md"], confidence=0.2)
    low_quality_outcome = AdvisorOutcome(
        run_id="run-low",
        status="partial",
        files_touched=["notes.txt"],
        retries=4,
        tests_run=[],
        review_verdict="weak targeting",
    )
    store.record_task_run(low_quality_packet, low_quality_advice, advisor_model="advisor-test", latency_ms=10, prompt_hash="low")
    store.record_outcome(low_quality_outcome)
    store.record_reward_label(
        compute_reward_label(
            low_quality_packet,
            low_quality_advice,
            low_quality_outcome,
            human_rating=0.0,
        )
    )

    out = tmp_path / "curated.jsonl"
    count = export_training_examples(store, out, min_quality_score=0.5)

    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert count == 2
    assert [row["run_id"] for row in rows] == ["run-positive", "run-negative"]
    assert rows[0]["example_type"] == "positive"
    assert rows[1]["example_type"] == "negative"
    assert rows[1]["hard_case_bucket"] == "constraint_failure"
    assert rows[0]["split"] == rows[1]["split"]
    assert rows[0]["reward_label"]["reward_profile_id"] == "legacy-generic"
