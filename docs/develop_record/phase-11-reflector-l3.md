# Phase 11 Reflector L3 / SelfModel

## Status

Completed as deterministic v1.

## Implemented

- Added scheduler-compatible `ReflectorL3` that aggregates current projections
  and event history, emits only `self_model_updated`, no-ops when unchanged, and
  applies a 12h emit throttle.
- Added L3 aggregators for capabilities, failure modes, preferred strategies,
  stable preferences, value tradeoffs, and role-level interaction patterns.
- Added `subject_view` and updated `SubjectProjection` so
  `SubjectProjection.current()` returns the latest persisted SelfModel plus the
  current ValueLens.
- Added CLI commands:
  - `alpha cognition self-model`
  - `alpha cognition self-model history --last N`
  - `alpha cognition reflect-l3 --once`
- Added focused tests for aggregators, L3 emission/throttle, SubjectProjection
  replay/current behavior, and CLI output.

## V1 Boundaries

- No LLM-based self-narration or semantic clustering.
- No direct L3 writes to beliefs, strategies, or lenses.
- No daemon-owned L3 cadence; the worker is scheduler-compatible and the CLI
  manual path calls it directly.
- Interaction patterns are role-level deterministic counters, not
  counterpart-specific beliefs.
