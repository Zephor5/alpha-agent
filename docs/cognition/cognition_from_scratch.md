# Cognition Runtime Architecture

This document describes the current cognition architecture.

Foreground behavior is owned by the runtime agent turn. The runtime allocates
one `turn_id`, persists session messages and runtime traces, calls the main LLM
and tools, then writes audit events that projections and background workers can
consume.

```text
runtime turn
  -> allocate AgentTurnContext
  -> append user session message
  -> record perceived input audit
  -> assemble prompt from session context and stable profile snapshot
  -> run the LLM/tool loop
  -> persist assistant and tool messages
  -> record acted and source-link audit events
  -> projections and workers materialize durable views
```

## Runtime Identity

`turn_id` is the foreground causal identity. It appears in persisted session
messages, runtime traces, tool metadata, memory proposal events, and turn audit
events.

`session_id` is the transcript and context bucket. It owns prompt history,
context-window foreground state, tool messages, runtime traces, and internal
self-signal streams.

## Event Vocabulary

Foreground runtime events are intentionally small:

- `perceived`: accepted runtime input.
- `acted`: completed model/tool-loop outcome.
- `turn_sources_recorded`: persisted artifacts linked to one turn.
- `memory_proposed`: audit record for a proposed memory update.
- `received_feedback`: explicit external or service feedback, not an automatic
  per-turn placeholder.

Belief lifecycle changes are not event-sourced belief state. Accepted memory
updates and retained workers mutate `atomic_beliefs` or `summary_beliefs`
directly; cognition events remain audit/source records.

## Memory

The main LLM decides when to call `memory_recall` or `memory_propose`.

`memory_recall` is read-only and returns compact belief handles with id,
content, type, scope, lifecycle, and held_since. Runtime does not perform hidden
dynamic recall from the user message.

`memory_propose` owns write gating for explicit memory updates. The model sends
an operation (`append_distinct`, `reinforce`, `replace`, `merge`, `correct`, or
`retract`) plus typed memory content when needed. `target_belief_ids` are
mutation targets; `reviewed_candidate_ids` records candidates reviewed before a
distinct append. Accepted updates mutate the belief lifecycle; uncertain updates
return candidates or require user confirmation instead of silently replacing
active beliefs. Tool results include `next_action` so the LLM can review
candidates or ask the user for confirmation without a separate review queue.

## Projections And Workers

Current cognition state keeps direct belief stores and a small retained
projection set:

- `atomic_beliefs`
- `summary_beliefs`
- `counterpart_view`
- `goal_view`
- `subject_view`

Daemon startup creates a `BackgroundCognitionService` when
`[cognition.background].enabled` is true. It shares the daemon's single
subject-level `LoopCoordinator` with all daemon-created `AlphaAgent` instances.
Background ticks are bounded gate checks: source intake, LLM memory extraction,
LLM consolidation, conflict review, and expired-belief archival run only when
their lower-layer source material is eligible. The timer decides when to check
for work; it does not refresh cognition by elapsed time alone.

Background progress uses the sidecar processing ledger rather than mutating raw
session messages or replaying audit logs. Summary-generation gates exist in
configuration, but profile/domain/self summary synthesis is still deferred.
Removed deterministic workers are not preserved as compatibility shims.

## Drive Loop

The Drive Loop produces normal runtime self-signal turns with stable internal
session ids. Goal progress is linked to the runtime turn audit events that were
actually written.
