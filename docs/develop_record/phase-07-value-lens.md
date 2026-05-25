# Phase 07 ValueLens Conflict Resolution v1

## Status

Implemented deterministic v1.

## Delivered

- Added `cognition/value/` with deterministic `ValueProfile` derivation,
  default/persisted `ValueLens` helpers, and a score-based conflict resolver.
- Belief materialization now derives a deterministic `ValueProfile` when an
  incoming belief record has no value weights; deterministic consolidation
  belief creation does the same.
- Added `subject_value_lens` storage and wired `SubjectProjection.current()` to
  return the persisted lens, with replay from `value_lens_shifted` events.
- Added `resolve_queued_conflicts` and `learn_value_lens` consolidation workers
  and registered them with the Phase 06 worker set.
- `resolve_queued_conflicts` consumes `consolidation_conflict_queued`, emits
  `belief_superseded` with decisive value kinds and lens explanation, and sends
  ties to `conflict_kept_for_human_review`.
- `learn_value_lens` implements a conservative sensitivity-only adjustment after
  repeated resolved tradeoffs, rate-limited to one shift per 24 hours using
  event-log order rather than event-id lexical order.
- `learn_value_lens` respects `last_processed_event_id` after successful/no-op
  runs. Yielded resumes start after the yielded metadata cursor and wrap only
  within the post-checkpoint event window.
- Added CLI commands:
  - `alpha cognition lens show [subject]`
  - `alpha cognition lens set [subject] --priority safety,honesty,...`
- Interpreter now records `proposed_resolution` for contradictions among
  recalled beliefs; durable supersede remains owned by the queued worker in v1.

## Deliberate v1 Boundaries

- The resolver uses the existing project `ValueKind` vocabulary:
  `safety`, `honesty`, `helpfulness`, `autonomy`, `efficiency`, and `learning`.
  Older draft names such as `existence`, `utility`, and `moral` are not used.
- Counterpart `trust_level` is not part of the v1 score. It remains a later
  resolver enhancement.
- Reviser does not directly emit durable `belief_superseded` during Reactive
  ticks. The durable path is the checkpointed queued-conflict worker.
- Lens learning only adjusts sensitivity. It does not reorder priority.

## Verification

Focused coverage lives in `tests/cognition/test_value_lens_phase07.py` and
covers profile derivation, resolver winner/tie behavior, lens-shaped supersede
direction, queued conflict handling, learning rate limit/sensitivity shift, CLI
show/set, SubjectProjection replay, empty-profile conflict derivation, and
non-chronological event-id ordering for the learning rate-limit check. Review
coverage also verifies that later learning windows do not re-count stale
supersede events after a successful checkpoint.

Final verification for this phase should include:

- `uv run ruff check .`
- `uv run pytest -q`
