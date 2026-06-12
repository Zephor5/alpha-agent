# Durable System Reminder Context Plan

## Status

Planned.

## Date

2026-06-12

## Goal

Unify session-stable prompt context around durable `system_reminder` source
messages, while keeping extraction from treating reminder-only content as new
memory evidence.

This plan focuses on self-memory summary and counterpart profile context. Local
time reminders are already turn-adjacent source-stream context; they are useful
temporal context for message timing, not memory extraction evidence and not
memory facts.

## Core Position

Stable session context should be visible in the same source stream that the
runtime replays.

The runtime currently has two different concepts:

- session source messages, which define ordered replay and extraction windows
- session summary snapshots, which record selected stable context but are
  rendered separately into prompts

Once durable `system_reminder` messages exist, self-memory and counterpart
profile context can use the same replay mechanism as other session-visible
context. The snapshot record can still exist as the selection ledger, but the
LLM-visible reminder should be materialized into `session_messages`.

## Target Model

### Selection Ledger

Keep a session-level record of which stable summary was selected:

- summary kind
- target kind and id
- source belief id
- selected content
- selected time

This record answers: "Which stable context did this session choose?"

### Source Stream Reminder

Materialize the selected context as durable `session_messages` rows:

- `kind = "system_reminder"`
- `llm_role = "user"`
- metadata identifies the reminder type:
  - `reminder_type = "self_memory_summary"`
  - `reminder_type = "counterpart_profile"`
- raw content uses the existing `<system-reminder>` wrapper

These rows answer: "Where did this stable context enter the session replay?"

The snapshot table remains the authoritative session-level selection record.
Stable-context reminder messages do not need to duplicate full snapshot
provenance metadata; they only need enough metadata to identify their reminder
type. Code that needs the selected belief or target can resolve it through the
session's snapshot record.

## Reminder Taxonomy And Retrieval Policy

`system_reminder` is a broad message kind. It is not specific enough for
retrieval decisions.

All code that reads reminder messages must filter by `metadata.reminder_type`
before applying subtype-specific logic. A query that only checks
`kind = "system_reminder"` is valid for broad replay, but not for semantic
selection.

Required reminder types:

- `session_time`: local calendar-time anchors for runtime inputs
- `self_memory_summary`: stable self-memory summary context selected for a
  session
- `counterpart_profile`: stable counterpart profile context selected for a
  session

Retrieval rules:

- Time reminder logic must find the latest reminder with
  `reminder_type = "session_time"`. It must not treat self-memory or
  counterpart-profile reminders as the latest time anchor.
- Stable-context reminder logic must find reminders by the concrete stable
  context type it needs: `self_memory_summary` or `counterpart_profile`.
- Broad source replay may include all `system_reminder` rows in ordinal order.
- Extraction may include all reminder rows as context, but evidence attribution
  must distinguish `<system-reminder>` context from ordinary source evidence.
- CLI, history, and debug display paths should filter out reminder messages
  directly. Reminder subtype filtering is an internal retrieval concern, not a
  display contract.
- Explicit low-level debug or source-inspection paths may show reminder
  messages when the caller is intentionally inspecting the raw source stream.

## Prompt Assembly Behavior

Prompt assembly should prefer replaying durable reminder messages from session
history instead of separately injecting summary/profile reminders from snapshot
records.

Expected order for a new session:

1. runtime system message
2. durable self-memory summary reminder, if selected
3. durable counterpart profile reminder, if selected
4. durable time reminder for the first runtime input
5. runtime input message

These stable summary/profile reminders are inserted only at session start. Later
turns replay them from `session_messages` in ordinal order. Compression does not
cause summary/profile reminders to be reinserted.

For this plan, session start means the moment before the first runtime input is
persisted for a session. The runtime performs stable summary/profile selection
once at that point. If no self-memory summary or counterpart profile is
available then, that session does not receive a later backfilled stable-context
reminder when background summaries appear or change.

## Compression Behavior

Compression may cover durable summary/profile reminders with ordinary source
messages. That is acceptable. The compressed handover becomes the continuity
record for earlier summary/profile reminder context.

The runtime should not materialize fresh summary/profile reminders after a
compression boundary. The selected snapshot remains the session-level selection
record, but durable summary/profile reminder insertion is a session-start action,
not a post-compression action.

Compression should not turn summary/profile reminders into user facts. They are
context-control messages.

## Extraction Behavior

Extraction prompts should explicitly define `<system-reminder>` semantics using
the model-visible wrapper, not database-only message kind terminology:

```text
Messages wrapped in <system-reminder>...</system-reminder> are session context,
not new user evidence. Use them only to interpret ordinary user, assistant, and
tool messages. Do not extract a new memory whose only support is a
<system-reminder> message.
```

This instruction should apply to both inactive backlog extraction and direct
compact extraction.

Extraction can include `system_reminder` messages in the prompt because they may
help interpret ordinary messages. The key rule is evidence attribution:

- a memory candidate may use `<system-reminder>` content as context
- a memory candidate must have direct support from at least one ordinary source
  message
- if an extraction window contains only `<system-reminder>` messages,
  extraction should yield no memory candidates

## Source Attribution Rule

New extracted memories should not be sourced only to `system_reminder` messages.

If program-side validation can identify source ids or source spans, it should
reject or ignore candidates whose only supporting source is a `system_reminder`.
If the first implementation does not have source-span-level validation, the
model-visible prompt instruction is still required, and tests should cover the
expected model contract with representative prompts.

## Implementation Tasks

- Add reminder metadata conventions for self-memory summary and counterpart
  profile durable reminders.
- Update existing time reminder retrieval so it selects only
  `reminder_type = "session_time"`, not every `system_reminder`.
- Add shared reminder lookup helpers that require a concrete `reminder_type`
  for semantic selection.
- Materialize selected self-memory summary and counterpart profile context into
  `session_messages` as `system_reminder` rows.
- Stop separately injecting summary/profile reminders from snapshots once
  durable session-start reminders exist.
- Keep snapshot records as the stable selection ledger.
- Insert summary/profile reminders only at session start; do not reinsert them
  after compression.
- Update answer prompt assembly to replay durable reminders in source ordinal
  order.
- Update inactive backlog extraction prompt construction so it has the same
  reminder semantics as direct compact extraction.
- Add extraction instruction text that distinguishes context reminders from
  extractable evidence.
- Add tests that reminder-only windows do not produce new memories.
- Add tests that ordinary user evidence can still be interpreted with reminder
  context.
- Add tests that compression does not reinsert summary/profile reminders.
- Add tests that mixed reminder types do not confuse time-anchor lookup,
  stable-context lookup, or extraction selection, including a case where a
  non-time reminder appears after the latest time reminder.
- Add CLI, history, and debug display filtering that hides reminder messages
  directly.
- Keep an explicit low-level debug or source-inspection path that can show raw
  reminder messages.

## Acceptance Checklist

- [ ] Self-memory summary context can appear as durable `system_reminder`
      session messages.
- [ ] Counterpart profile context can appear as durable `system_reminder`
      session messages.
- [ ] Session summary snapshots remain the stable selection ledger.
- [ ] Summary/profile reminders are inserted at session start only.
- [ ] Answer prompts do not duplicate the same summary/profile context through
      both snapshot injection and source-message replay.
- [ ] Time reminder lookup ignores non-time `system_reminder` messages.
- [ ] Stable-context reminder lookup ignores `session_time` reminders.
- [ ] CLI, history, and debug display paths filter out reminder messages
      directly.
- [ ] Explicit low-level debug or source-inspection paths can still show raw
      reminder messages.
- [ ] Inactive backlog extraction and direct compact extraction use consistent
      `system_reminder` semantics.
- [ ] Extraction prompt text uses `<system-reminder>` and says reminder messages
      are context, not new evidence.
- [ ] Reminder-only extraction windows produce no new memory candidates.
- [ ] Extracted memories are not sourced only to `system_reminder` messages.
- [ ] Compression does not reinsert summary/profile reminders after the
      boundary.

## Non-Goals

- Do not remove session summary snapshots unless a replacement selection ledger
  exists.
- Do not change the LLM role representation away from user-role
  `<system-reminder>` messages in this work.
- Do not treat local time reminders as memory facts.
