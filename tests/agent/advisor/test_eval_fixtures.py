import json

from agent.advisor.evaluation.eval_fixtures import EvalFixture, load_eval_fixture


def test_load_eval_fixture_roundtrip(tmp_path):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "fixture_id": "image-ui-hero",
                "domain": "image-ui",
                "description": "hero refresh from updated screenshot",
                "input_packet": {
                    "run_id": "run-1",
                    "task_text": "refresh the hero from the updated screenshot",
                    "task_type": "ui-update",
                    "repo": {"path": "/tmp/ui", "branch": "main", "dirty": False},
                    "repo_summary": {"modules": ["ui"], "hotspots": ["ui/home.png"], "file_tree_slice": ["ui/home.png"]},
                    "candidate_files": [{"path": "ui/home.png", "reason": "changed screenshot", "score": 0.9}],
                    "recent_failures": [],
                    "constraints": ["preserve spacing scale"],
                    "tool_limits": {"image_read": True},
                    "acceptance_criteria": ["hero matches updated mock"],
                    "token_budget": 700,
                    "task": {"domain": "image-ui", "text": "refresh the hero from the updated screenshot", "type": "ui-update"},
                    "context": {"summary": "image-ui task context", "metadata": {"changed_files": ["ui/home.png"]}},
                    "artifacts": [{"kind": "image", "locator": "ui/home.png", "description": "latest mock", "metadata": {"changed": True}, "score": 1.0}],
                    "history": [],
                    "domain_capabilities": [
                        {
                            "domain": "image-ui",
                            "supported_artifact_kinds": ["image"],
                            "supported_packet_fields": [
                                "task",
                                "context",
                                "artifacts",
                                "constraints",
                                "history",
                                "acceptance_criteria",
                            ],
                            "supports_changed_artifacts": True,
                            "supports_symbol_regions": True,
                        }
                    ]
                },
                "expected_advice": {
                    "focus_targets": ["ui/home.png"],
                    "anti_targets": ["docs/brief.md"],
                    "required_plan_steps": ["inspect ui/home.png"],
                    "forbidden_plan_steps": ["broad refactor"],
                    "expected_failure_modes": ["spacing drift"]
                },
                "human_review_rubric": {
                    "scale": [0, 1, 2, 3],
                    "criteria": ["helpfulness", "oversteer", "calibration"]
                },
            }
        ),
        encoding="utf-8",
    )

    fixture = load_eval_fixture(fixture_path)

    assert isinstance(fixture, EvalFixture)
    assert fixture.fixture_id == "image-ui-hero"
    assert fixture.input_packet.task.domain == "image-ui"
    assert fixture.expected_advice.focus_targets == ["ui/home.png"]
    assert fixture.human_review_rubric.criteria == ["helpfulness", "oversteer", "calibration"]
