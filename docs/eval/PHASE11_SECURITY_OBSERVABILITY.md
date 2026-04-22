# Phase 11 security, privacy, and observability

Phase 11 adds the first security/privacy/observability layer around live Advisor runs.

## What exists now

The repo now provides:
- safe-default packet redaction via `redact_packet()`
- structured JSONL run-event logging via `RunEventLogger`
- live metrics export via `export_live_metrics()`
- audit reporting via `build_audit_report()`
- retention and hosted-deployment settings:
  - `retention_days`
  - `event_log_path`
  - `redact_sensitive_fields`
  - `hosted_mode`

## Stored data expectations

Advisor now explicitly documents and exports the main stored surfaces:
- trace database
- event log
- packets
- advice records
- run outcomes
- reward labels
- run manifests and run lineage records

Retention is configuration-driven through `retention_days`.

## Safe defaults

`redact_packet()` applies conservative redaction for exported packet views:
- email addresses -> `[REDACTED:email]`
- token/secret/password fragments -> `[REDACTED:secret]`
- session/task/user/tenant identifiers -> `[REDACTED:id]`

This is meant for exports, audits, and operator tooling, not for mutating canonical stored runs.

## Structured event logs

`AdvisorOrchestrator` now emits run-scoped structured events for:
- `run.started`
- `routing.decided`
- `executor.completed`
- `verifier.completed`
- `reward.recorded`
- `run.completed`

Each event includes:
- timestamp
- event type
- run id
- stage
- payload

## Metrics export

`export_live_metrics()` summarizes:
- total runs
- lineage-backed runs
- reward-labeled runs
- success/failure/partial counts
- advisor vs baseline arm counts
- verifier status counts

This provides the first metrics export path for both local evaluation and live runs.

## Audit report intent

`build_audit_report()` combines:
- stored data surfaces
- retention expectations
- redaction defaults
- dataset provenance summary
- reward-label lineage coverage
- metrics snapshot

Use this as the operator-facing audit note for current deployments.

## Hosted / multi-tenant operating note

Phase 11 does not add full tenancy controls, but it does define the minimum operating posture:
- use dedicated `ADVISOR_HOME` per deployment or tenant slice
- keep trace DB and event log paths isolated per deployment
- enable redaction in exported packet views by default
- treat event logs and trace DBs as retention-managed operator data
- rotate/archive artifacts on the configured retention boundary

## What later phases can assume

Later phases can now build on:
- stable run-scoped event logs
- retention-aware runtime settings
- redacted export surfaces
- audit-ready provenance summaries
- basic hosted-mode operating guidance
