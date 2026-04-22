import json
import subprocess
import sys
import textwrap
import time

from agent.advisor.api import create_orchestrator
from agent.advisor.integrations import (
    BuildTestCommandVerifier,
    CodingAgentSubprocessExecutor,
    DomainWorkerSubprocessExecutor,
    FrontierHTTPExecutor,
    HumanReviewFileVerifier,
    IntegrationRegistry,
    RubricTextVerifier,
    ScreenshotHashVerifier,
)
from agent.advisor.orchestration import DeterministicABRouter
from agent.advisor.schemas import AdviceBlock, AdvisorInputPacket, CandidateFile, RepoSummary
from agent.advisor.settings import AdvisorSettings
from agent.advisor.trace_store import AdvisorTraceStore


class StubRuntime:
    def generate_advice(self, packet, system_prompt=None):
        return AdviceBlock(
            task_type=packet.task_type,
            relevant_files=[{"path": "main.py", "why": "entrypoint", "priority": 1}],
            recommended_plan=["inspect main.py"],
            confidence=0.92,
        )


def _packet(run_id: str = "run-phase12"):
    return AdvisorInputPacket(
        run_id=run_id,
        task_text="repair the main flow",
        task_type="bugfix",
        repo={"path": "/tmp/repo", "branch": "main", "dirty": False, "session_id": "sess-12"},
        repo_summary=RepoSummary(modules=["app"], hotspots=["main.py"], file_tree_slice=["main.py"]),
        candidate_files=[CandidateFile(path="main.py", reason="token overlap", score=0.9)],
        recent_failures=[],
        constraints=["tests must pass"],
        tool_limits={"terminal": True},
        acceptance_criteria=["repair succeeds"],
        token_budget=900,
    )


def test_frontier_http_executor_posts_canonical_payload_and_normalizes_response(tmp_path):
    transcript = tmp_path / "request.json"
    server_script = tmp_path / "http_server.py"
    server_script.write_text(
        textwrap.dedent(
            f"""
            import json
            from http.server import BaseHTTPRequestHandler, HTTPServer

            transcript = {str(transcript)!r}

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers.get('content-length', '0'))
                    body = self.rfile.read(length)
                    with open(transcript, 'wb') as handle:
                        handle.write(body)
                    response = {{
                        'status': 'success',
                        'summary': 'frontier executor completed',
                        'output': 'patched main.py',
                        'files_touched': ['main.py'],
                        'tests_run': ['pytest -q'],
                        'metadata': {{'provider': 'stub-http'}},
                    }}
                    encoded = json.dumps(response).encode('utf-8')
                    self.send_response(200)
                    self.send_header('content-type', 'application/json')
                    self.send_header('content-length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

                def log_message(self, format, *args):
                    return

            server = HTTPServer(('127.0.0.1', 8765), Handler)
            server.handle_request()
            """
        )
    )

    proc = subprocess.Popen([sys.executable, str(server_script)])
    try:
        time.sleep(0.2)
        executor = FrontierHTTPExecutor(name="frontier-http", endpoint_url="http://127.0.0.1:8765")
        result = executor.execute_request(_packet())
    finally:
        proc.wait(timeout=5)

    sent = json.loads(transcript.read_text(encoding="utf-8"))
    assert sent["packet"]["run_id"] == "run-phase12"
    assert result.status == "success"
    assert result.files_touched == ["main.py"]
    assert result.metadata["provider"] == "stub-http"


def test_subprocess_executors_normalize_json_output(tmp_path):
    script = tmp_path / "executor.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            payload = json.loads(sys.stdin.read())
            response = {
                "status": "partial",
                "summary": f"handled {payload['packet']['task_type']}",
                "output": payload.get("rendered_advice"),
                "files_touched": ["main.py"],
                "tests_run": [],
                "metadata": {"executor": payload["packet"]["run_id"]},
            }
            print(json.dumps(response))
            """
        )
    )

    coding = CodingAgentSubprocessExecutor(name="coding-subprocess", command=["python", str(script)])
    worker = DomainWorkerSubprocessExecutor(name="worker-subprocess", command=["python", str(script)])

    coding_result = coding.execute_request(_packet("run-coding"), rendered_advice="hint")
    worker_result = worker.execute_request(_packet("run-worker"), rendered_advice=None)

    assert coding_result.status == "partial"
    assert coding_result.output == "hint"
    assert coding_result.metadata["executor"] == "run-coding"
    assert worker_result.metadata["executor"] == "run-worker"


def test_verifier_integrations_cover_build_rubric_screenshot_and_human_review(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pass.py").write_text("print('ok')\n", encoding="utf-8")
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    expected.write_bytes(b"same-bytes")
    actual.write_bytes(b"same-bytes")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({"run-phase12": {"status": "pass", "summary": "review approved"}}), encoding="utf-8")

    packet = _packet()
    packet.repo["path"] = str(repo)

    build = BuildTestCommandVerifier(name="pytest-check", command=["python", "pass.py"])
    rubric = RubricTextVerifier(name="rubric-check", required_phrases=["patched", "main.py"])
    screenshot = ScreenshotHashVerifier(name="screenshot-check")
    human = HumanReviewFileVerifier(name="human-review", review_file=reviews)

    executor_result = CodingAgentSubprocessExecutor(
        name="coding-inline",
        command=[
            "python",
            "-c",
            "import json; print(json.dumps({'status':'success','summary':'patched main.py','output':'patched main.py','files_touched':['main.py'],'tests_run':[]}))",
        ],
    ).execute_request(packet, rendered_advice="hint")

    build_result = build.verify_request(packet, executor_result)
    rubric_result = rubric.verify_request(packet, executor_result)
    screenshot_result = screenshot.verify_artifacts(run_id=packet.run_id, expected_path=expected, actual_path=actual)
    human_result = human.verify_request(packet, executor_result)

    assert build_result.status == "pass"
    assert rubric_result.status == "pass"
    assert screenshot_result.status == "pass"
    assert human_result.summary == "review approved"


def test_integration_registry_builds_real_components_and_preserves_baseline_advisor_parity(tmp_path):
    script = tmp_path / "executor.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            payload = json.loads(sys.stdin.read())
            print(json.dumps({
                'status': 'success',
                'summary': 'patched main.py',
                'output': payload.get('rendered_advice'),
                'files_touched': ['main.py'],
                'tests_run': ['pytest -q'],
                'metadata': {'saw_advice': payload.get('rendered_advice') is not None},
            }))
            """
        )
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pass.py").write_text("print('ok')\n", encoding="utf-8")
    store = AdvisorTraceStore(tmp_path / "advisor.db")
    settings = AdvisorSettings(enabled=True, trace_db_path=str(tmp_path / "advisor.db"), event_log_path=str(tmp_path / "events.jsonl"))
    registry = IntegrationRegistry()
    executor = registry.create_executor(
        {
            "kind": "coding_agent_subprocess",
            "name": "coding-subprocess",
            "command": ["python", str(script)],
        }
    )
    verifier = registry.create_verifier(
        {
            "kind": "build_test_command",
            "name": "build-check",
            "command": ["python", "pass.py"],
        }
    )

    baseline = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=executor,
        verifiers=[verifier],
        router=DeterministicABRouter(advisor_fraction=0.0),
    )
    advisor = create_orchestrator(
        settings=settings,
        runtime=StubRuntime(),
        trace_store=store,
        executor=executor,
        verifiers=[verifier],
        router=DeterministicABRouter(advisor_fraction=1.0),
    )

    baseline_packet = _packet("run-baseline-parity")
    advisor_packet = _packet("run-advisor-parity")
    baseline_packet.repo["path"] = str(repo)
    advisor_packet.repo["path"] = str(repo)

    baseline_result = baseline.run(baseline_packet)
    advisor_result = advisor.run(advisor_packet)

    assert baseline_result.manifest.executor == advisor_result.manifest.executor
    assert baseline_result.manifest.verifiers == advisor_result.manifest.verifiers
    assert baseline_result.lineage.executor_result.metadata["saw_advice"] is False
    assert advisor_result.lineage.executor_result.metadata["saw_advice"] is True
    assert baseline_result.lineage.executor_result.files_touched == advisor_result.lineage.executor_result.files_touched
