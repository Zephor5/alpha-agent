# Phase 08 Reflector L2 Control v1

## Status

Implemented deterministic v1.

## Delivered

- Added `StrategyOverride` and SQLite-backed `StrategyProjection` over
  `strategy_view`.
- Added scheduler-compatible `ReflectorL2` with four deterministic rules:
  recurring contradiction reflections, feedback surprise streak, lens shift
  flap, and premature novel auto-form burst.
- `feedback-surprise-streak` requires five consecutive misses for the same
  trigger; a success for that trigger resets the streak.
- `lens-shift-flap` groups repeated shifts by a deterministic direction key from
  before/after priority or sensitivity deltas.
- `StrategyProjection` enforces the active non-expired strategy cap at
  materialization time, so replay cannot produce more than five concurrent
  active overrides.
- Added active strategy application in Reactive flow:
  - `require_confirm_before_novel_form` marks novel interpretations as needing
    confirmation.
  - `disable_auto_procedure_match_for_trigger` skips procedure matching for the
    matching trigger.
  - `require_explicit_confirm_on_contradiction` emits
    `belief_form_pending_confirmation` during revise.
  - `freeze_lens_learning_for_24h` makes `learn_value_lens` skip.
- Added `expire_strategies` consolidation worker and registered it with default
  consolidation workers.
- Added CLI:
  - `alpha cognition strategies --active`
  - `alpha cognition strategies --all`
  - `alpha cognition strategy-expire <id>`

## Deliberate v1 Boundaries

- No free-form strategy DSL.
- No semantic clustering for L2 rules.
- No daemon-owned L2 scheduler. `ReflectorL2` is scheduler-compatible and can
  run under the shared scheduler with `LoopPriority.L2`, but Phase 08 does not
  introduce a long-running background daemon loop.
- Reviser does not synthesize a full belief candidate in the pending
  confirmation event; it records the strategy reason and contradiction refs.
- Drive-loop self-signal counterpart matching remains deferred until Drive Loop
  exists.

## Verification

Focused coverage lives in `tests/cognition/test_reflector_l2_phase08.py` and
covers L2 rules, strategy projection/expiry, counterpart matching, Reactive
strategy application, lens-learning freeze, procedure-match suppression, and CLI
listing/manual expiry.

Final verification for this phase should include:

- `uv run ruff check .`
- `uv run pytest -q`
