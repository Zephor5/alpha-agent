# Phase 06 Consolidation Loop v1

## Status

Implemented deterministic v1.

## Delivered

- Added `cognition/loops/` with scheduler/checkpoint primitives,
  `Scheduler.tick()`, `ConsolidationLoop.run_once()`, worker reports, and six
  workers.
- Added SQLite tables for `cognition_worker_checkpoint`,
  `context_window_background`, and `procedure_view`.
- Completed `context_compressed` handling in `ContextWindowProjection`: absorbed
  foreground ids are removed unless anchored, and summaries are stored in
  `context_window_background`.
- Replaced the procedure stub with a minimal SQLite-backed projection that
  handles learned/strengthened/weakened procedure events and deterministic
  trigger matching.
- Added CLI: `alpha cognition consolidate --now [--dry-run]`.
- CLI dry-run executes against a temporary SQLite clone, copies SQLite `-wal`
  and `-shm` sidecars when present, and leaves the real DB unchanged.

## Deliberate v1 Boundaries

- No real background daemon scheduler. The scheduler is in-process and reusable,
  but only `run_once()` is wired for CLI/tests.
- Procedure learning is deterministic and conservative: same decision pattern
  plus successful feedback count, not semantic clustering.
- Context compression uses deterministic concatenation/truncation, not LLM
  summarization.
- Conflict queuing is not consumed in Phase 06; ValueLens remains the Phase 07
  owner.
- Workers use resumable checkpoint records with deterministic cursor metadata.
  A yielded checkpoint bypasses `min_interval`, and missing cursor items resume
  at the first sorted item greater than the cursor. v1 chunking is per worker,
  counterpart, thread, belief, or grouped pattern rather than long-running
  streaming batches.

## Verification

Focused coverage lives in `tests/cognition/test_consolidation_loop.py` and
covers worker idempotency, context compression, counterpart digest supersede and
replay, checkpoint persistence/resume, scheduler tick gating and backlog cursor
updates, procedure replay, config rewrite preservation, and CLI dry-run DB
immutability. The second-review patch also covers yielded checkpoint immediate
resume, disappeared-cursor resume, and no-op scan yields.

Final local verification:

- `uv run ruff check .` passed.
- `uv run pytest -q` passed with 219 tests.
- Repository text scan found no local machine-specific absolute paths in
  README, AGENTS, config example, docs, source, or tests.
