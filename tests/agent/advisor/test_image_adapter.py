from agent.advisor.image_adapter import ImageUIContextAdapter
from agent.advisor.schemas import CandidateFile, FailureSignal


def test_image_adapter_builds_packet_from_ui_artifacts():
    adapter = ImageUIContextAdapter(token_budget=800)

    packet = adapter.build_packet(
        run_id="run-ui-1",
        task_text="align the hero with the updated mock and screenshot",
        task_type="ui-update",
        repo={"path": "/tmp/ui", "branch": "main", "dirty": False},
        file_tree_slice=[
            "ui/mockups/home.png",
            "ui/layouts/home.json",
            "references/brand-guide.pdf",
        ],
        candidate_files=[
            CandidateFile(path="ui/mockups/home.png", reason="latest mock", score=0.8),
            CandidateFile(path="ui/layouts/home.json", reason="current layout", score=0.7),
            CandidateFile(path="references/brand-guide.pdf", reason="brand reference", score=0.5),
        ],
        recent_failures=[
            FailureSignal(
                kind="recent-failure",
                file="ui/layouts/home.json",
                summary="previous pass drifted from the header region",
                fix_hint="anchor changes to the hero region",
            )
        ],
        constraints=["preserve spacing scale"],
        tool_limits={"image_read": True},
        acceptance_criteria=["hero matches mock"],
        changed_files=["ui/mockups/home.png"],
    )

    assert packet.task.domain == "image-ui"
    assert packet.context.summary == "image-ui task context"
    assert [artifact.kind for artifact in packet.artifacts] == ["image", "layout", "reference"]
    assert packet.artifacts[0].metadata["changed"] is True
    assert packet.history[0].metadata["fix_hint"] == "anchor changes to the hero region"
    assert packet.domain_capabilities[0].domain == "image-ui"
    assert packet.domain_capabilities[0].supported_artifact_kinds == ["image", "layout", "reference"]
    assert packet.domain_capabilities[0].supports_changed_artifacts is True


def test_image_adapter_prefers_changed_images_and_layouts_over_references():
    adapter = ImageUIContextAdapter(token_budget=800)

    packet = adapter.build_packet(
        run_id="run-ui-2",
        task_text="refresh the hero image and layout",
        task_type="ui-update",
        repo={"path": "/tmp/ui", "branch": "main", "dirty": False},
        file_tree_slice=["references/brand-guide.pdf", "ui/layouts/home.json", "ui/mockups/home.png"],
        candidate_files=[
            CandidateFile(path="references/brand-guide.pdf", reason="brand guide", score=0.9),
            CandidateFile(path="ui/layouts/home.json", reason="layout config", score=0.8),
            CandidateFile(path="ui/mockups/home.png", reason="updated mock", score=0.7),
        ],
        recent_failures=[],
        constraints=[],
        tool_limits={},
        acceptance_criteria=[],
        changed_files=["ui/mockups/home.png"],
    )

    assert [candidate.path for candidate in packet.candidate_files] == [
        "ui/mockups/home.png",
        "ui/layouts/home.json",
        "references/brand-guide.pdf",
    ]


def test_image_adapter_exposes_region_hooks_when_task_targets_a_ui_area():
    adapter = ImageUIContextAdapter(token_budget=800)

    packet = adapter.build_packet(
        run_id="run-ui-3",
        task_text="align the hero region with the updated screenshot",
        task_type="ui-update",
        repo={"path": "/tmp/ui", "branch": "main", "dirty": False},
        file_tree_slice=["ui/mockups/home.png"],
        candidate_files=[CandidateFile(path="ui/mockups/home.png", reason="updated mock", score=0.8)],
        recent_failures=[],
        constraints=[],
        tool_limits={"image_read": True},
        acceptance_criteria=[],
        changed_files=["ui/mockups/home.png"],
    )

    assert packet.domain_capabilities[0].supports_symbol_regions is True
    assert packet.artifacts[0].metadata["region_hint"] == "hero"
