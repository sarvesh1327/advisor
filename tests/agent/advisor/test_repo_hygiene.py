from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_contributing_guide_exists():
    contributing = REPO_ROOT / "CONTRIBUTING.md"
    assert contributing.exists()
    text = contributing.read_text(encoding="utf-8")
    assert "virtualenv" in text.lower()
    assert "pytest tests/agent/advisor -q" in text


def test_ci_workflow_exists_and_tests_supported_python_versions():
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "python-version" in text
    assert "3.11" in text
    assert "3.12" in text
    assert "pytest tests/agent/advisor -q" in text


def test_release_workflow_exists():
    workflow = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "build" in text.lower()


def test_pyproject_includes_ruff_and_supported_python_metadata():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "requires-python = \">=3.11\"" in pyproject
    assert "[tool.ruff]" in pyproject
    assert "[tool.ruff.lint]" in pyproject


def test_python_version_file_exists():
    python_version = REPO_ROOT / ".python-version"
    assert python_version.exists()
    assert python_version.read_text(encoding="utf-8").strip() == "3.12"
