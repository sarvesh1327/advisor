# Phase 15 — Operator surface and deployment path

## Goal
Make Advisor operable as a real system with explicit deployment guidance, inspectable live state, resumable long-running work, and operational retention enforcement.

## What was added

### Operator runtime
- `agent/advisor/operator_runtime.py`
- deployment profiles for:
  - `single_tenant`
  - `hosted`
- operator snapshot builder for:
  - deployment metadata
  - live run metrics
  - run summaries with lineage/reward provenance
  - benchmark summary rollups
  - queued job state

### Background jobs
- persistent JSON-backed operator job queue
- job lifecycle states:
  - `queued`
  - `running`
  - `completed`
  - `failed`
- resumability for failed/running jobs that carry a `resume_token`

### Retention enforcement
- retention enforcer archives stale runs into JSONL
- archived run payloads include lineage links for auditability
- stale event-log lines rotate into archived JSONL files
- archived runs are deleted from the live SQLite store after archival

### Operator API surface
Added HTTP routes in `create_app(...)`:
- `GET /v1/operator/overview`
- `GET /v1/operator/runs/{run_id}`
- `GET /v1/operator/jobs`
- `POST /v1/operator/jobs`
- `POST /v1/operator/jobs/{job_id}/resume`
- `POST /v1/operator/retention/enforce`

### Operator CLI surface
Added CLI commands:
- `advisor operator-overview`
- `advisor deployment-profile --mode hosted`
- `advisor retention-enforce`

## Audit / paper-faithful guardrails
- operator snapshot is read-only over canonical run state
- provenance links remain attached through:
  - run id
  - reward-label presence
  - lineage presence
- benchmark summaries are computed from immutable benchmark manifests passed into the snapshot builder
- deployment/operator metadata is kept separate from canonical benchmark manifests

## Verification

### Red
- `python -m pytest tests/agent/advisor/test_operator_runtime.py -q`
- failed first with:
  - `ModuleNotFoundError: agent.advisor.operator_runtime`

### Green targeted
- `python -m pytest tests/agent/advisor/test_operator_runtime.py tests/agent/advisor/test_api.py tests/agent/advisor/test_cli.py -q`
- `11 passed`

### Lint
- `ruff check .`
- passed

### Full suite
- `python -m pytest tests/agent/advisor -q`
- `96 passed`

## Done criteria coverage
- deployable single-tenant and hosted paths: yes
- run inspection/dashboard surface: yes
- background jobs, queueing, resumability: yes
- retention/archival/rotation operational enforcement: yes
