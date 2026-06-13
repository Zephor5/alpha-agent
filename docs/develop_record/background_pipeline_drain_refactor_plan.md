# Background Pipeline Drain Refactor Plan

## Status
Planned. Do not implement from this document until explicitly requested.

## Objective
Refactor daemon-owned background cognition so a tick performs a singleflight, until-idle pipeline drain instead of running each eligible worker once. The new pipeline must improve large import catch-up throughput, simplify stage scheduling, remove unused intake and legacy consolidation-loop paths, and keep recovery semantics simple through idempotent ledger state.

## Agreed Design
- `BackgroundCognitionService` is the only daemon background pipeline entry point.
- A tick is singleflight. If one tick is running, another scheduled or wake-triggered tick does not start a second drain.
- A tick may run until backlog is empty. There is no wall-clock tick timeout.
- `LoopCoordinator` remains the cross-loop cooperative lock. The refactor removes the old trigger/watch scheduler model, not foreground/background coordination.
- Internal coordinator chunk TTL and daemon shutdown wait remain bounded implementation constants. They are not background tick deadlines and are not user-facing config.
- Import success wakes background processing from the daemon runtime layer. If a tick is already running, the wake is a no-op.
- Any worker/window failure records an error and aborts the current tick. Retry happens on the next interval or external wake.
- Yield or stop requested aborts the current tick only before a unit is claimed. Once a unit is claimed or a stage run is started, that unit must finish successfully or fail.
- Daemon/background startup recovers all abandoned ledger work by marking `CLAIMED` source progress/windows and `STARTED` stage runs `FAILED`. `PENDING` remains valid backlog.
- `skipped_no_backlog` means only "no backlog". Missing LLM providers, configuration errors, and invariant mismatches are `error`.
- LLM workers process one natural unit per invocation. The service drains backlog by repeated invocation; removed batch/min knobs are not retained as worker constructor parameters or private tuning constants.

## Pipeline Semantics
Each tick loops until one full pass makes no progress:

```text
repeat:
  extraction: drain up to max_sessions_per_pass complete eligible sessions ordered by sessions.created_at
  consolidation: drain all pending/retryable backlog
  conflict_review: drain all pending/retryable backlog
  summary: drain all eligible summary targets
  archive: archive all expired beliefs
  stop when no stage made progress
```

Extraction details:
- Eligible sessions are determined only by inactivity gate and retryable extraction source presence.
- `active_session_ids` filtering is removed.
- pending handover filtering is removed.
- ordinary and import sessions share scheduling and candidate construction.
- sessions are ordered by durable `sessions.created_at`, with stable tie-breaks, through an explicit `StateStore` session-record listing API.
- one selected session is fully drained before moving to the next selected session.
- `max_sessions_per_pass` limits how many complete sessions extraction drains before moving downstream.
- background backlog extraction keeps the latest compressed-message boundary rule.
- direct compact extraction remains a separate path for hot-prefix cache reuse.
- all extraction paths, including direct compact extraction, never emit session-scope memory. `BeliefScope.SESSION` remains in the global model for non-extraction paths.
- import prompts keep their runtime-context restriction; ordinary prompts may keep compressed/runtime context.

Work unit definitions:
- extraction: one session-targeted extraction window; service drains one selected session until no extraction backlog remains.
- consolidation: one pending extracted draft.
- conflict review: one pending conflict window.
- summary: one eligible summary target.
- archive: all expired beliefs in one local pass.

## Configuration Shape
Keep only active background settings:

```toml
[cognition.background]
enabled = true
startup_delay_seconds = 5
interval_seconds = 300

[cognition.background.extraction]
inactivity_threshold_hours = 24
max_sessions_per_pass = 10

[cognition.background.summary]
initial_min_beliefs = 12
changed_source_min = 6
invalidated_source_min = 1
```

Remove:
- `cognition.consolidation.*`
- `cognition.background.tick_timeout_seconds`
- `cognition.background.intake.*`
- `cognition.background.extraction.min_sources`
- `cognition.background.consolidation.*`
- `cognition.background.conflict.*`
- `cognition.background.summary.batch_size`
- worker `dry_run` config
- worker batch/min constructor parameters that only mirror removed background config

Internal non-config constants may remain for cooperative lock TTL and daemon shutdown wait only. They must not reintroduce a wall-clock tick budget or hidden worker batch sizing.

## Non-Goals
- No compatibility logic for existing persisted data.
- No `docs/` updates outside this `docs/todo/` plan unless explicitly requested later.
- No pipeline-owned LLM rate limiter in the first implementation.
- No session-internal chunking for background extraction.
- No replacement CLI command for the removed `cognition consolidate` command.
- No global removal of `BeliefScope.SESSION`.
- No coordination with other `docs/todo/` plans. This document is evaluated against current active code when implemented.

## Phase 1: Remove Legacy Scheduling and Intake

### Task 1.1: Remove intake stage from background cognition
**Description:** Delete the source intake worker, intake config, intake helper functions, and intake-specific tests. Extraction already scans source messages directly, so intake ledger marking is not part of the target pipeline.

**Acceptance criteria:**
- [ ] `BackgroundStage.INTAKE` is removed.
- [ ] `SourceIntakeWorker` and intake helper functions are removed.
- [ ] background service no longer mentions intake or passes intake config.
- [ ] tests no longer assert intake progress rows.

**Likely files:**
- `src/alpha_agent/cognition/processing_ledger.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/__init__.py`
- `src/alpha_agent/config.py`
- `tests/cognition/test_consolidation_loop.py`
- `tests/test_config.py`

### Task 1.2: Remove legacy `ConsolidationLoop` and CLI command
**Description:** Delete the old synchronous consolidation loop and the `alpha cognition consolidate` command. The daemon background service becomes the only background pipeline entry point.

**Acceptance criteria:**
- [ ] `ConsolidationLoop` and `ConsolidationConfig` are removed.
- [ ] `cognition consolidate` CLI command and helper functions are removed.
- [ ] `cognition.consolidation.*` config fields and environment mappings are removed.
- [ ] imports and exports no longer reference the old loop.

**Likely files:**
- `src/alpha_agent/cognition/loops/consolidation.py`
- `src/alpha_agent/cognition/loops/__init__.py`
- `src/alpha_agent/cli.py`
- `src/alpha_agent/config.py`
- `tests/test_config.py`

### Task 1.3: Remove trigger-based scheduler abstractions
**Description:** Remove the old trigger/watch scheduler model. Keep `LoopCoordinator` as the cross-loop cooperative lock, and keep only worker reporting, checkpoint, and cooperative-yield primitives needed by background service and direct services. `DriveLoop` behavior is not part of this background drain refactor; if shared scheduler types are deleted, replace any drive-only type usage locally without changing drive semantics.

**Acceptance criteria:**
- [ ] `Scheduler`, `InMemoryCheckpointStore`, background worker `ScheduleTrigger`, and `handles_event_kinds`/`trigger` requirements are removed.
- [ ] workers no longer define trigger metadata.
- [ ] `WorkerCheckpoint.last_processed_event_id` is removed from model and checkpoint storage.
- [ ] checkpoint storage still records `worker_name`, `last_run_at`, `last_status`, and `metadata`.
- [ ] `LoopCoordinator` remains in use by daemon foreground turns, background drain, direct compact extraction coordination where applicable, and DriveLoop.
- [ ] DriveLoop tests and behavior remain unchanged.

**Likely files:**
- `src/alpha_agent/cognition/loops/scheduler.py`
- `src/alpha_agent/cognition/loops/drive.py`
- `src/alpha_agent/cognition/loops/workers/*.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- tests that construct or assert worker checkpoints

### Task 1.4: Clean configuration surface
**Description:** Remove obsolete background config keys and add `extraction.max_sessions_per_pass`. Replace `tick_timeout_seconds` consumers with internal implementation constants for cooperative chunk TTL and daemon shutdown wait.

**Acceptance criteria:**
- [ ] config dataclasses match the agreed configuration shape.
- [ ] config load, set, list, env var, and deprecated-key logic no longer expose removed settings.
- [ ] daemon shutdown and direct-service shutdown no longer read `cognition.background.tick_timeout_seconds`.
- [ ] `LoopAcquireRequest.max_chunk_duration` uses an internal cooperative chunk TTL constant, not a tick deadline.
- [ ] `config.example.toml` and `.env.example` match the new shape.

**Likely files:**
- `src/alpha_agent/config.py`
- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/compact_extraction.py`
- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- `config.example.toml`
- `.env.example`
- `tests/test_config.py`

## Phase 2: Implement Singleflight Pipeline Drain

### Task 2.1: Replace one-shot eligible worker execution with pipeline drain
**Description:** Rework `BackgroundCognitionService.tick_once()` so it loops through fixed stage order until no stage makes progress, failure occurs, or cooperative yield/stop is requested before the next unit is claimed.

**Acceptance criteria:**
- [ ] one tick can run extraction, downstream stages, and return to extraction repeatedly.
- [ ] one full no-progress pass exits normally.
- [ ] worker/window error aborts the tick and records background error.
- [ ] yield/stop before a unit is claimed aborts the tick without background error.
- [ ] once a unit is claimed or a stage run is started, the unit finishes successfully or fails; no claimed unit is left as a normal yielded result.
- [ ] tick has no wall-clock deadline; cooperative chunk TTL is used only by `LoopCoordinator`.
- [ ] tick singleflight behavior is preserved.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `tests/cognition/test_consolidation_loop.py`

### Task 2.2: Add startup recovery for abandoned claimed work
**Description:** On background service startup, recover the shared processing ledger by marking all `CLAIMED` source windows and source progress as `FAILED`, and marking all `STARTED` stage runs as `FAILED`.

**Acceptance criteria:**
- [ ] startup recovery does not touch `PENDING`.
- [ ] recovery is stage-agnostic.
- [ ] recovery includes `BackgroundStage.FEEDBACK_ATTRIBUTION` rows as well as daemon background pipeline stages.
- [ ] recovered rows can be retried by normal pipeline logic.
- [ ] recovery writes a clear error reason such as `recovered abandoned claimed background work`.

**Likely files:**
- `src/alpha_agent/cognition/processing_ledger.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `tests/cognition/test_consolidation_loop.py`
- `tests/cognition/test_feedback_attribution.py`

### Task 2.3: Wake background processing after import
**Description:** Trigger a non-blocking background wake from daemon runtime after a successful non-dry-run conversation import. Do not implement pending-wake behavior, and do not make `ConversationImportService` depend on background runtime.

**Acceptance criteria:**
- [ ] successful non-dry-run import with inserted messages wakes background processing.
- [ ] dry run does not wake background processing.
- [ ] if a tick is already running, wake does not start another tick.
- [ ] import response is not blocked by a full drain.
- [ ] direct uses of `ConversationImportService` remain pure import/write operations.

**Likely files:**
- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `tests/test_daemon_runtime.py`

### Task 2.4: Use backlog predicates instead of worker skip probes
**Description:** Service-level stage drivers check backlog counts before invoking workers. Workers also keep strict status semantics: missing required LLM providers are `error`, and `skipped_no_backlog` means only no backlog.

**Acceptance criteria:**
- [ ] downstream stages do not call workers just to discover no backlog.
- [ ] missing provider is `error` in extraction, consolidation, conflict review, summary, and direct compact extraction paths.
- [ ] service may preflight missing provider before worker invocation when backlog is known.
- [ ] worker skip with positive backlog is treated as invariant error.
- [ ] normal no-backlog completion does not save a skipped checkpoint.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/loops/workers/memory_summary.py`
- worker tests for no-provider behavior

## Phase 3: Refactor Extraction Selection and Semantics

### Task 3.1: Move session selection to `BackgroundCognitionService`
**Description:** Service selects eligible sessions ordered by `sessions.created_at`, then calls extraction worker for a specified session until that session has no backlog.

**Acceptance criteria:**
- [ ] `StateStore` exposes an explicit session-record listing API ordered by `sessions.created_at ASC, session_id ASC`.
- [ ] service selects up to `max_sessions_per_pass` sessions per extraction pass.
- [ ] selected sessions use the new durable `sessions.created_at` ordering API, not `list_session_ids()` ordering.
- [ ] service drains one session to no backlog before moving to the next.
- [ ] normal per-session `skipped_no_backlog` reports are not included in tick reports and do not overwrite checkpoint status.

**Likely files:**
- `src/alpha_agent/state/store.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `tests/cognition/test_consolidation_loop.py`

### Task 3.2: Add session-targeted extraction worker entry point
**Description:** Add a worker method that processes one extraction window for a specified session. The worker remains responsible for candidate construction, LLM call, validation, and ledger writes for that one unit. Direct compact extraction remains a separate explicit job entry point, but follows the same extraction validation contract.

**Acceptance criteria:**
- [ ] `run_session_once(session_id)` or equivalent exists.
- [ ] it returns `ok`, `skipped_no_backlog`, `yielded`, or `error` with tightened semantics.
- [ ] it does not choose among global sessions.
- [ ] direct compact extraction still uses the compact-job entry point and remains independent from daemon backlog scheduling.
- [ ] missing LLM provider returns `error` from both session and compact-job extraction paths.

**Likely files:**
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `src/alpha_agent/cognition/loops/compact_extraction.py`
- tests for direct compact extraction and backlog extraction

### Task 3.3: Unify ordinary and import backlog candidates
**Description:** Remove import-specific scheduling and candidate selection. All sessions use one backlog candidate builder with latest compressed-message boundary and retryable extraction source refs.

**Acceptance criteria:**
- [ ] ordinary/import session scheduling is unified.
- [ ] latest compressed boundary is honored for all sessions.
- [ ] system reminders and compressed messages are excluded as extraction sources.
- [ ] import session prompts still avoid ordinary runtime identity/context.

**Likely files:**
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- tests for import extraction prompt and ordering

### Task 3.4: Remove redundant eligibility gates
**Description:** Remove `active_session_ids` and pending-handover gates from background extraction eligibility.

**Acceptance criteria:**
- [ ] background service no longer accepts or passes `active_session_ids`.
- [ ] extraction worker no longer filters by active session ids.
- [ ] pending handover traces no longer block background extraction.
- [ ] eligibility is inactivity gate plus retryable source presence.
- [ ] old tests expecting active-session or pending-handover extraction skips are removed or inverted to match the target rule.

**Likely files:**
- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- related tests

### Task 3.5: Ban session scope for extraction
**Description:** Remove session refs from extraction allowed refs and prompt instructions, and enforce the rule in extraction validation. This applies to ordinary backlog, import backlog, and direct compact extraction. Keep `BeliefScope.SESSION` in the global model for non-extraction paths.

**Acceptance criteria:**
- [ ] extraction-stage validation rejects session-scope outputs for all extraction source paths.
- [ ] prompt instructions tell the LLM not to emit session scope.
- [ ] ordinary backlog, import backlog, and direct compact extraction all follow this rule.
- [ ] allowed-about refs for extraction never include `("session", session_id)`.
- [ ] summary behavior is not directly changed.

**Likely files:**
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `src/alpha_agent/cognition/background_llm_contract.py`
- tests for extraction validation, prompts, and direct compact extraction

## Phase 4: Downstream Drain and Reporting

### Task 4.1: Drain consolidation until empty
**Description:** Repeatedly run consolidation while pending/retryable extracted draft backlog exists. Each worker invocation processes one pending extracted draft as its natural unit.

**Acceptance criteria:**
- [ ] all consolidation backlog is drained before conflict review stage starts.
- [ ] one consolidation worker invocation selects one pending extracted draft.
- [ ] removed consolidation batch/min knobs are not retained as constructor parameters or private batch sizing constants.
- [ ] worker errors abort the tick.
- [ ] worker skip with positive backlog is treated as error.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- consolidation tests

### Task 4.2: Drain conflict review until empty
**Description:** Repeatedly run conflict review while pending/retryable conflict windows exist. Each worker invocation processes one pending conflict window.

**Acceptance criteria:**
- [ ] all conflict review backlog is drained before summary starts.
- [ ] one conflict review worker invocation selects one pending conflict window.
- [ ] removed conflict batch/min knobs are not retained as constructor parameters or private batch sizing constants.
- [ ] `PENDING` conflict windows remain legal backlog.
- [ ] failures abort the tick and are retried by a future tick.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- conflict review tests

### Task 4.3: Drain summary targets until naturally converged
**Description:** Repeatedly run summary worker while summary target predicate finds work. Each worker invocation processes one eligible summary target.

**Acceptance criteria:**
- [ ] summary targets are drained until predicate returns zero.
- [ ] one summary worker invocation selects one eligible summary target.
- [ ] summary batch size fields and constructor parameters are removed.
- [ ] normal summary supersede behavior converges.
- [ ] summary failures abort the tick.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/memory_summary.py`
- summary worker tests

### Task 4.4: Keep archive as final all-items local stage
**Description:** Archive expired beliefs at the end of each pass. Keep archive worker behavior as a local all-items operation.

**Acceptance criteria:**
- [ ] archive stage runs after summary.
- [ ] archive remains local and does not add per-item caps.
- [ ] archive progress contributes to pass progress.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/workers/archive_expired.py`

### Task 4.5: Preserve raw unit reports and checkpoint observability
**Description:** Keep tick reports as the list of actual work/error/yield reports. Do not include normal no-backlog probe reports.

**Acceptance criteria:**
- [ ] repeated worker invocations can appear as repeated reports.
- [ ] successful work saves checkpoint.
- [ ] errors/yields save checkpoint.
- [ ] normal no-backlog checks do not overwrite last successful status.

**Likely files:**
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/cognition/loops/scheduler.py` or its successor module
- tests around report counts and checkpoint status

## Phase 5: Tests, Config Examples, and README

### Task 5.1: Update import progress tests
**Description:** Preserve message-level import status counts for extraction pending/processed/failed.

**Acceptance criteria:**
- [ ] import status still reports extraction progress per imported message.
- [ ] multi-message extraction windows mark all covered messages processed or failed together.
- [ ] import wake behavior is covered.
- [ ] wake is asserted at daemon runtime boundary; `ConversationImportService` remains runtime-agnostic.

**Likely files:**
- `src/alpha_agent/state/store.py`
- `src/alpha_agent/daemon/runtime.py`
- `tests/test_daemon_runtime.py`
- `tests/test_conversation_import_service.py`

### Task 5.2: Update background pipeline tests
**Description:** Replace one-worker-per-tick expectations with until-idle drain expectations.

**Acceptance criteria:**
- [ ] one tick can process multiple sessions.
- [ ] extraction rotates downstream after `max_sessions_per_pass`.
- [ ] failure aborts and next tick can retry.
- [ ] yield/stop before claim aborts without background error.
- [ ] claim/start followed by failure leaves retryable `FAILED` ledger state.
- [ ] startup recovery unblocks claimed rows.
- [ ] startup recovery includes feedback attribution claimed progress.
- [ ] no-provider is error in every LLM worker and direct compact extraction path.
- [ ] `skipped_no_backlog` is not saved for normal service-level no-backlog checks.

**Likely files:**
- `tests/cognition/test_consolidation_loop.py`
- `tests/cognition/test_feedback_attribution.py`
- focused new tests if splitting the file improves clarity

### Task 5.3: Update config and CLI tests
**Description:** Align tests with removed config keys, removed command, and new extraction max sessions setting.

**Acceptance criteria:**
- [ ] removed config keys cannot be listed/set as active settings.
- [ ] new `max_sessions_per_pass` setting loads from config/env.
- [ ] removed `cognition consolidate` command is no longer expected.
- [ ] removed tick timeout config no longer appears in config show/list/env tests.
- [ ] daemon shutdown tests do not rely on background tick timeout config.

**Likely files:**
- `tests/test_config.py`
- `tests/test_cli_daemon.py`
- `tests/test_cli_agent_loop.py`

### Task 5.4: Update root documentation and examples
**Description:** Update active root-level user/config files only.

**Acceptance criteria:**
- [ ] README no longer documents removed consolidation/intake behavior if present.
- [ ] `config.example.toml` matches new config shape.
- [ ] `.env.example` matches new env var surface.
- [ ] no files under `docs/` are updated except this plan unless explicitly requested.

**Likely files:**
- `README.md`
- `config.example.toml`
- `.env.example`

## Final Verification
Run the current project validation gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

Additional targeted checks:
- [ ] Import a multi-conversation payload and verify background wake starts processing.
- [ ] Verify one tick drains more than one session.
- [ ] Verify session selection follows `sessions.created_at ASC, session_id ASC`.
- [ ] Verify a synthetic failed extraction aborts the tick and retries on the next tick.
- [ ] Verify direct compact extraction still runs independently.
- [ ] Verify all extraction paths, including direct compact extraction, reject session-scope outputs.
- [ ] Verify missing LLM provider is an error for extraction, consolidation, conflict review, summary, and direct compact extraction.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Until-idle tick runs for a long time on huge backlogs | Background thread stays busy | Singleflight prevents overlap; stop/yield is checked before each claim; extraction rotates downstream every `max_sessions_per_pass` sessions |
| Failed LLM output repeats on every interval | Repeated errors/API cost | Failure aborts the tick and records status; no immediate retry loop |
| Removing scheduler abstractions breaks hidden imports | Build/test failures | Keep `LoopCoordinator`, handle DriveLoop locally without behavior changes, use `rg` before deletion, and rely on full mypy/pytest gate |
| Import/ordinary unification changes prompt behavior unexpectedly | Memory extraction quality regression | Keep import runtime-context restriction; add prompt tests |
| Removing session scope changes expected memory outputs | Test and behavior updates required | Scope ban applies to all extraction paths only; global scope enum remains for non-extraction paths |
| Removing worker sizing knobs creates unbounded LLM windows | Oversized prompts or slow calls | Worker unit definitions stay one natural unit; extraction session backlog is drained by repeated windows rather than hidden batch constants |

## Open Questions
- None. Implementation should not proceed until explicitly requested.
