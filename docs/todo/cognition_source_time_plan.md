# Cognition Source Time Execution Plan

## Status

Planned.

## Date

2026-06-12

## Scope

This plan makes cognition use source message time at the decision points where
the time of the evidence matters. It does not add a persistent evidence-time
field to belief models, and it does not change the meanings of existing belief
time fields.

Only this todo document is updated in this planning pass. Do not update
`docs/cognition/` as part of this plan unless a later implementation task asks
for it explicitly.

## Goal

Cognition must distinguish three concepts:

- Processing time: when Alpha records an event, runs a worker, writes a ledger
  row, asks an LLM to interpret evidence, or writes an audit record.
- Holding time: when Alpha starts holding or accepting a belief in cognition
  state.
- Source message time: when the supporting session message actually happened.

Processing and holding time remain useful and should not be removed. Source
message time should be resolved from `session_messages.created_at` only where
cognition makes semantic decisions about recency, supersession, feedback, or
temporal user claims.

## Decisions

- Persisted timestamps are UTC. Incoming timestamp values are normalized to UTC
  before persistence.
- User-facing and prompt-facing source times are rendered in the session's local
  timezone.
- Add a session-level timezone record. Prefer IANA timezone names such as
  `Asia/Shanghai`; allow fixed offsets only when an external source can provide
  nothing better.
- CLI/runtime sessions use the local IANA timezone when first created. If the
  local IANA name cannot be discovered without adding a dependency, use the
  current local fixed offset.
- Gateway sessions may receive a timezone from external metadata. Missing
  timezone falls back to the local IANA timezone. Invalid explicit timezone is
  rejected.
- A session timezone is chosen when the session is created. Later messages do
  not update it, compare against it, or warn on mismatch.
- Internal self-signal and goal sessions also get a session record and use the
  same local timezone fallback rule.
- `AtomicBelief.held_since` and `SummaryBelief.held_since` keep meaning "when
  Alpha started holding this belief."
- `ValidityWindow.observed_at` keeps meaning "when cognition/LLM observed the
  evidence or formed this judgment."
- No new persistent evidence-time field is added to atomic or summary beliefs.
- Source message time is resolved dynamically from belief/session sources when
  it is needed.
- `memory_recall` output and ranking are unchanged in this plan.
- Direct `memory_propose` remains out of scope for source-time changes.
- Runtime traces, ledgers, audits, LLM traces, daemon status, background status,
  and worker checkpoints continue to use processing time.

## Current Behavior

Most cognition writes currently use UTC processing time:

- Cognitive events default to the emitter clock.
- Background extraction, consolidation, and summary acceptance write
  `held_since` and default `validity.observed_at` from worker processing time.
- Direct memory proposals use the emitter clock.
- Feedback attribution records feedback at the generated feedback event time.
- Extraction, consolidation, and summary prompts do not expose source message
  time.

Session messages already have `created_at`, but the value is not validated as a
UTC invariant at all write boundaries and is not consistently propagated into
cognition decisions.

## Target Semantics

### Processing And Holding Time

Keep processing time for operational records:

- background source progress rows
- background source windows and stage runs
- worker checkpoints
- runtime traces
- LLM call traces
- daemon/background status timestamps
- audit row creation time

Keep holding time for belief state:

- `held_since`
- lifecycle transition `at` fields
- default `validity.observed_at` when a worker accepts an LLM output

These fields describe Alpha's execution and belief lifecycle, not the source
message's original occurrence time.

### Source Message Time

For cognition derived from session messages, resolve source message time from
supporting `session_message` references:

- `PERCEIVED.timestamp` uses the input user message `created_at`.
- `ACTED.timestamp` uses the assistant message `created_at`.
- `RECEIVED_FEEDBACK.timestamp` uses the feedback user message `created_at`.
- Feedback history uses the feedback user message `created_at` for same-day
  de-duplication.
- Extraction prompts receive one source-window local time line.
- Consolidation prompts receive one local source-time line per draft/active
  belief record.
- Summary prompts receive one local source-time line per source belief record.
- Consolidation recency decisions prefer source message time over holding time.

System reminders and compressed handover messages are context, not evidence.
They must not contribute to source-time ranges.

## Prompt Time Format

Store source-time ranges in metadata as UTC ISO strings. Render them into
prompts using the session timezone.

For one source instant:

```text
Source message time: 2026-06-12 09:00 (Asia/Shanghai).
```

For a range:

```text
Source message time range: 2026-06-12 09:00 to 2026-06-12 09:17 (Asia/Shanghai).
```

Do not include UTC strings or extra structured timezone fields in prompts.

## Implementation Tasks

### Task 1: Add Session Timezone State

**Description:** Add a durable session-level record so prompt rendering can use
one timezone source for CLI, gateway, and internal sessions.

**Acceptance criteria:**

- [ ] A session record stores `session_id`, `timezone`, `created_at`, and
      `updated_at`.
- [ ] Timezone values prefer valid IANA names.
- [ ] Fixed offsets are accepted only as a fallback representation.
- [ ] Invalid explicit timezone values are rejected.
- [ ] CLI/runtime-created sessions use the local IANA timezone when available
      and the current local fixed offset otherwise.
- [ ] Internal self-signal and goal sessions create session records with the
      same local timezone fallback rule.
- [ ] Gateway-created sessions use external timezone metadata when valid and
      the local timezone fallback rule when missing.
- [ ] Later messages do not update or compare session timezone.

**Verification:**

- [ ] Tests cover CLI default timezone creation.
- [ ] Tests cover gateway timezone from metadata.
- [ ] Tests cover gateway missing timezone fallback.
- [ ] Tests cover invalid explicit gateway timezone rejection.
- [ ] Tests cover internal goal/self-signal session timezone creation.

**Files likely touched:**

- `src/alpha_agent/state/schema.sql`
- `src/alpha_agent/state/models.py`
- `src/alpha_agent/state/store.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/gateway/session.py`
- gateway bridge files that build `source_metadata`
- tests under `tests/` and `tests/cognition/`

### Task 2: Normalize Session Message Timestamps

**Description:** Make `session_messages.created_at` and `updated_at` reliable UTC
invariants at the state-store boundary.

**Acceptance criteria:**

- [ ] `append_session_message(created_at=None)` continues to use the current UTC
      time.
- [ ] Explicit `created_at` and `updated_at` values must be non-empty,
      parseable datetimes.
- [ ] Naive datetimes are treated as UTC before persistence.
- [ ] Offset-aware datetimes are normalized to UTC before persistence.
- [ ] `insert_session_message()` rejects empty or invalid `created_at`.
- [ ] `insert_session_message()` validates `updated_at` when present.
- [ ] No public shared timestamp helper is introduced unless multiple modules
      genuinely need it.

**Verification:**

- [ ] Tests cover default timestamp auto-fill.
- [ ] Tests cover explicit offset timestamp normalization to UTC.
- [ ] Tests cover naive timestamp normalization to UTC.
- [ ] Tests cover empty and invalid timestamp rejection for append and insert.

**Files likely touched:**

- `src/alpha_agent/state/store.py`
- `tests/test_session_context.py` or a focused state-store test module

### Task 3: Resolve And Render Source Time Ranges

**Description:** Add program-owned helpers for resolving source message time from
session-message refs and rendering prompt-facing local time lines.

**Acceptance criteria:**

- [ ] Resolver supports `session_message` refs only.
- [ ] Belief source-time range resolution reads `belief.sources` and uses only
      `Reference("session_message", id)` sources.
- [ ] `system_reminder` messages are skipped for evidence ranges.
- [ ] `compressed_message` messages are skipped for evidence ranges.
- [ ] Non-session refs are ignored, not treated as source time.
- [ ] If a selected session-message evidence ref is missing or has an invalid
      timestamp, the caller fails fast.
- [ ] Local prompt rendering uses the stored session timezone.
- [ ] Single-instant and range prompt lines match the formats in this document.

**Verification:**

- [ ] Tests cover source-window range from multiple messages.
- [ ] Tests cover single-message prompt line.
- [ ] Tests cover reminder and compressed messages being excluded.
- [ ] Tests cover local timezone rendering from session state.

**Files likely touched:**

- `src/alpha_agent/cognition/loops/workers/_common.py` or a new cognition helper
- `src/alpha_agent/state/store.py`
- `tests/cognition/test_consolidation_loop.py`

### Task 4: Align Foreground Event Timestamps

**Description:** Timestamp source-representing cognition events with the
timestamped source message they represent.

**Acceptance criteria:**

- [ ] `PERCEIVED.timestamp` equals `user_record.created_at`.
- [ ] `ACTED.timestamp` equals `assistant_record.created_at`.
- [ ] `TURN_SOURCES_RECORDED.timestamp` remains the emitter/processing time.
- [ ] Counterpart observation/identification events remain processing time.
- [ ] Runtime traces and LLM traces remain processing time.

**Verification:**

- [ ] Tests cover a delayed or explicit `created_at` user message and verify
      `PERCEIVED.timestamp`.
- [ ] Tests verify `ACTED.timestamp` follows the assistant message.
- [ ] Tests verify bookkeeping events still use processing time.

**Files likely touched:**

- `src/alpha_agent/runtime/agent.py`
- `tests/test_agent_loop.py`
- `tests/cognition/test_drive_loop_behavior.py`

### Task 5: Add Extraction Source-Time Context

**Description:** Give background extraction one source-window local time line
without modifying stable replay prefixes.

**Acceptance criteria:**

- [ ] `candidate.prompt_prefix_messages` is unchanged.
- [ ] `handover_prompt_prefix_hash` behavior is unchanged.
- [ ] Extraction instruction includes a source-window time line rendered in the
      session timezone.
- [ ] Source-window metadata stores UTC `source_time_start`,
      `source_time_end`, and `source_time_basis`.
- [ ] Source-window metadata excludes reminders and compressed messages from
      range calculation.
- [ ] No `extraction_version` bump is required.
- [ ] Accepted beliefs continue to write `held_since` and default
      `validity.observed_at` from processing/acceptance time.

**Verification:**

- [ ] Tests verify extraction prompt contains the local source-time line.
- [ ] Tests verify prompt prefix hash is unchanged by source-time rendering.
- [ ] Tests verify metadata UTC range is written.
- [ ] Tests verify reminder-only extraction still skips.
- [ ] Tests verify accepted belief `held_since` remains processing time.

**Files likely touched:**

- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- source-time helper from Task 3
- `tests/cognition/test_consolidation_loop.py`
- `tests/test_context_handover.py`

### Task 6: Add Consolidation Source-Time Context

**Description:** Let consolidation compare draft and active beliefs using
source-time ranges resolved from their session-message sources.

**Acceptance criteria:**

- [ ] Consolidation prompt records include `held_since`.
- [ ] Consolidation prompt records include a local source-time line when
      source-message evidence exists.
- [ ] Prompt rules state that recency decisions prefer source message time over
      `held_since`.
- [ ] Prompt rules state that `held_since` is Alpha holding time, not evidence
      time.
- [ ] Supersede, retract, and archive decisions must not infer source recency
      from `held_since`.
- [ ] Consolidation acceptance still writes `held_since`,
      `validity.observed_at`, lifecycle transition `at`, and audits using
      processing/acceptance time.

**Verification:**

- [ ] Tests verify draft and active belief prompt records include local
      source-time lines.
- [ ] Tests cover old imported/source evidence processed today not superseding
      newer source evidence just because `held_since` is newer.
- [ ] Tests verify accepted consolidated belief timing fields remain processing
      time.

**Files likely touched:**

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/state_service.py`
- `tests/cognition/test_consolidation_loop.py`

### Task 7: Use Feedback Message Time

**Description:** Make feedback attribution use the feedback user's source
message time for feedback events and same-day feedback history.

**Acceptance criteria:**

- [ ] `FeedbackAttributionJob` carries `user_message_created_at`.
- [ ] Runtime job submission passes `user_record.created_at`.
- [ ] `RECEIVED_FEEDBACK.timestamp` equals `user_message_created_at`.
- [ ] `record_belief_feedback(..., at=...)` receives
      `user_message_created_at`.
- [ ] Same-day feedback de-duplication uses feedback message time.
- [ ] Conflict-review window metadata includes `user_message_created_at`.
- [ ] Feedback attribution ledger and audits remain processing time.

**Verification:**

- [ ] Tests cover feedback processing delayed after the feedback message and
      verify event/history timestamps use message time.
- [ ] Tests cover same-day de-duplication based on feedback message date.
- [ ] Tests verify audit `created_at` remains processing time.

**Files likely touched:**

- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- `src/alpha_agent/cognition/state_service.py`
- `tests/cognition/test_feedback_attribution.py`
- `tests/cognition/test_feedback_loop_end_to_end.py`

### Task 8: Add Summary Source-Time Context

**Description:** Let summary prompts show the source-time range of each source
belief without changing summary belief storage semantics.

**Acceptance criteria:**

- [ ] Summary source belief records include `held_since`.
- [ ] Summary source belief records include a local source-time line.
- [ ] Summary prompt instructions avoid presenting old source evidence as newly
      updated evidence.
- [ ] Summary acceptance still writes `held_since` and default
      `validity.observed_at` from processing/acceptance time.
- [ ] Summary gate logic remains based on source belief ids, lifecycle, changed
      count, and invalidation count. It is not converted into a time gate.

**Verification:**

- [ ] Tests verify summary prompt records include local source-time lines.
- [ ] Tests verify summary belief timing fields remain processing time.
- [ ] Tests verify summary gating behavior is unchanged.

**Files likely touched:**

- `src/alpha_agent/cognition/loops/workers/memory_summary.py`
- `tests/cognition/test_self_memory_summary_worker.py`
- `tests/cognition/test_domain_summary_worker.py`

## Out Of Scope

- No persistent `evidence_time` field on beliefs.
- No semantic change to `held_since`.
- No semantic change to `validity.observed_at`.
- No `memory_recall` ranking or output change.
- No direct `memory_propose` source-time change.
- No runtime trace source-time support.
- No automatic timezone update from later gateway messages.
- No update to `docs/cognition/` in this planning pass.

## Acceptance Checklist

- [ ] All persisted timestamps touched by this plan are stored as UTC.
- [ ] Sessions have a durable timezone value.
- [ ] Prompt-facing source times render in the session timezone.
- [ ] `held_since` remains holding time across write paths.
- [ ] `validity.observed_at` remains cognition observation time across write
      paths.
- [ ] Source message time is available to extraction, consolidation, summary,
      and feedback through program-owned resolution.
- [ ] Reminder and compressed messages are not treated as source evidence.
- [ ] `PERCEIVED`, `ACTED`, and `RECEIVED_FEEDBACK` use their source message
      timestamps.
- [ ] Feedback history uses feedback message time.
- [ ] Consolidation prompts can compare incoming and existing beliefs using
      explicit source-time context.
- [ ] Background ledgers, audits, traces, checkpoints, and status timestamps
      remain processing time.
- [ ] Tests cover delayed background processing where source message time and
      processing time differ.
