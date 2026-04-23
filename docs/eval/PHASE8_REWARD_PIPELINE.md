# Phase 8 reward pipeline

Phase B upgrades the live reward path from a generic weighted component score to a profile-aware scalar reward registry while keeping legacy reward rows readable.

## Canonical reward inputs

Live reward is now computed from:
- canonical input packet
- canonical advice block
- executor outcome / executor metadata
- verifier results / verifier metadata
- optional constraint-violation list
- resolved `advisor_profile_id`

Human rating is no longer part of the live optimization path.

## Profile-aware reward registry

The live system resolves the active advisor profile first, then dispatches reward computation through a reward registry.

Current profile/spec mapping in the default config:
- `coding-default` -> `coding_swe_efficiency`
- `image-ui` -> `ui_from_text_layout`

The registry also supports additional built-in specs used by tests and later phases:
- `coding_exact_answer`
- `ui_edit_from_screenshot`
- `research_writing_match`

## Scalar reward formulas in the live path

### `coding_swe_efficiency`
- `R = 0` if unresolved
- `R = 0.5 + 0.5 * (MAX_STEPS - steps) / MAX_STEPS` if resolved
- `MAX_STEPS = 40`

### `ui_from_text_layout`
- `R = 0` if render/build is invalid
- otherwise `R = 0.75 * hard_constraint_pass_rate + 0.25 * soft_style_score`

Additional built-in formulas are available for later profile slices:
- `coding_exact_answer`
- `ui_edit_from_screenshot`
- `research_writing_match`

## Reward labels in the trace store

Reward labels are still persisted per run in SQLite as JSON, but new rows now carry profile-aware reward provenance:
- `advisor_profile_id`
- `reward_profile_id`
- `reward_formula`
- `reward_version`
- `raw_reward`
- `total_reward`
- `quality_score`
- `reward_diagnostics`
- `dataset_split`
- `example_type`
- `hard_case_bucket`
- `notes`

Legacy Phase 8 reward rows remain readable. They continue to surface:
- `reward_profile_id = legacy-generic`
- `reward_formula = weighted_components`
- legacy `components` diagnostics

## Export and curation rules

`export_training_examples()` still exports only runs with reward labels.

Curation behavior remains:
- filters out low-quality neutral examples with `min_quality_score`
- keeps negative examples even when their score is low
- assigns a stable split from repo-family + task-family when no explicit split override is given
- dedupes near-identical examples before writing JSONL
- preserves hard-case buckets for later focused replay or training

## Hard-case buckets

Current buckets remain lightweight and deterministic:
- `constraint_failure`
- `failed_execution`
- `targeting_miss`
- `inefficient_execution`

## Feedback-loop notes

Current live loop:
1. record packet/advice/outcome
2. resolve `advisor_profile_id`
3. compute scalar reward through the reward registry
4. persist reward provenance and diagnostics
5. export curated JSONL later for offline dataset / training work

Phase B intentionally does **not** yet make dataset manifests or checkpoint promotion profile-local. That lands in later follow-up phases.
