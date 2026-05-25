# Phase 10 Drive Loop v1

## Status

Completed as a deterministic synchronous v1.

## Implemented

- Added first-class `Goal` / `GoalId` model exports and SQLite-backed
  `GoalProjection` over `goal_view`.
- Added `GoalRegistry` for `goal_set`, `goal_satisfied`, `goal_abandoned`, and
  `goal_progressed` write paths.
- Registered `GoalProjection` in the default projection registry.
- Added `DriveLoop.run_once()` with default-disabled config, active goal
  selection, per-goal cooldown, cognition-thread `self_signal` creation, and
  `goal_progressed` drive metadata after successful Reactive ticks.
- Added `[cognition.drive]` config defaults, environment overrides, and nested
  config writer preservation for both consolidation and drive sections.
- Added CLI commands for goal list/set/satisfy/abandon and manual
  `alpha cognition drive --once`.
- Covered registry/projection rebuild, DriveLoop self-signal/cooldown/disabled
  behavior, CLI flow, and config preservation/env overrides.

## V1 Boundaries

- No daemon-owned Drive scheduler is started in this phase.
- No autonomous or LLM-based goal creation is implemented.
- One manual DriveLoop pass emits at most one self-signal.
- Goal satisfaction remains explicit through the registry/CLI.
