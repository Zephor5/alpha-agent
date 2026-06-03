# Cognition Loops

Phase 06 adds a synchronous, in-process Consolidation Loop v1. It exposes
`Scheduler.tick()` for due-worker runs, `ConsolidationLoop.run_once()` for
forced manual/test runs, and `alpha cognition consolidate --now` for CLI
operation. There is no background daemon scheduler in v1.

## Workers

- `merge_beliefs`: supersedes equivalent active beliefs with the highest
  confidence/latest survivor.
- `archive_expired`: archives active beliefs whose JSON applicability has a
  `valid_until` earlier than the current UTC time.
- `compress_context`: moves old unanchored foreground perception ids into
  `context_window_background` once foreground exceeds `context_foreground_max`.
- `summarize_counterpart`: maintains one active digest belief per counterpart
  after `counterpart_digest_min_beliefs`, refreshing it after
  `counterpart_digest_min_new_beliefs` new source beliefs.

All workers checkpoint through `cognition_worker_checkpoint` and call the loop
coordinator at worker/chunk boundaries, including no-op units. A checkpoint with
`last_status="yielded"` bypasses `min_interval` on the next `Scheduler.tick()`
so partial work resumes immediately. Workers that can yield store deterministic
cursor metadata. Resume scans process the sorted suffix after the cursor first,
then wrap around to the lower/equal prefix before completion, relying on worker
idempotency so work created during a yield window is not skipped before watched
event cursors advance.

CLI dry-runs operate on a temporary SQLite clone, including `-wal` and `-shm`
sidecar files when present, and do not write events, checkpoints, or projection
rows to the real database.

## Drive Loop

Phase 10 adds a synchronous Drive Loop v1. It is disabled by default through
`[cognition.drive].enabled = false`, but `alpha cognition drive --once` runs one
manual pass with `force=True`.

Goals are stored in `goal_view` from `goal_set`, `goal_satisfied`,
`goal_abandoned`, and `goal_progressed` events. `GoalRegistry` is the write path
for user/external goals, and `GoalProjection` rebuilds deterministically from
the event log. The active goal cap is enforced at both the registry and
projection boundary.

`DriveLoop.run_once()` selects one active goal by priority descending, then
least-recently driven first, with stable `updated_at`/id tie breaks. It acquires
the coordinator at `LoopPriority.DRIVE` only while selecting the goal and
creating a `self_signal`; it releases DRIVE before trying to acquire REACTIVE.
If REACTIVE is busy, the self-signal is dropped and the goal remains eligible
for a later pass. On a successful runtime self-signal turn, the loop emits
`goal_progressed` with `drive_progress=true`, which updates `last_drive_at` and
enforces per-goal cooldown.

There is no daemon-owned Drive scheduler in v1. The loop exposes a trigger shape
for later scheduler integration, while tests and CLI call `run_once()` directly.
