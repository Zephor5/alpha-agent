# Cognition Loops

Phase 06 adds a synchronous, in-process Consolidation Loop v1. It exposes
`Scheduler.tick()` for due-worker runs, `ConsolidationLoop.run_once()` for
forced manual/test runs, and `alpha cognition consolidate --now` for CLI
operation. There is no background daemon scheduler in v1.

## Workers

- `promote_judgment`: promotes repeated deterministic judgment claims within
  `judgment_repeat_window` when count reaches `judgment_repeat_threshold`.
- `merge_beliefs`: supersedes equivalent active beliefs with the highest
  confidence/latest survivor.
- `archive_expired`: archives active beliefs whose JSON applicability has a
  `valid_until` earlier than the current UTC time.
- `learn_procedure`: learns a minimal deterministic procedure when the same
  decision pattern has `procedure_success_threshold` successful feedback
  matches.
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
