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
- `memory_proposed`: proposed memory state transition.
- `belief_form_pending_confirmation`: concrete memory proposal requiring user
  confirmation.
- `received_feedback`: explicit external or service feedback, not an automatic
  per-turn placeholder.
- `reflected`: concrete audit findings based on runtime/tool outcomes.

Belief lifecycle events such as `belief_formed`, `belief_superseded`, and
`belief_retracted` are durable state events owned by tools and background
workers.

## Memory

The main LLM decides when to call `memory_recall` or `memory_propose`.

`memory_recall` is read-only and returns compact belief handles with id,
content, type, scope, and status. Runtime does not perform hidden dynamic recall
from the user message.

`memory_propose` owns write gating for explicit memory updates. The model sends
an operation (`append`, `reinforce`, `replace`, `merge`, `correct`, or
`retract`) plus typed memory content when needed. Accepted updates mutate the
belief lifecycle; uncertain updates return target candidates or require user
confirmation instead of silently replacing active beliefs.

## Projections And Workers

Projections keep durable, deterministic views over the event log:

- `belief_view`
- `context_window_view`
- `counterpart_view`
- `reflection_view`
- `strategy_view`
- `goal_view`
- `subject_view`

Background workers merge equivalent beliefs, archive expired state, compress
old foreground context, maintain counterpart summaries, resolve queued
conflicts, learn conservative value-lens shifts, expire strategies, and
aggregate a deterministic self-model.

## Drive Loop

The Drive Loop produces normal runtime self-signal turns with stable internal
session ids. Goal progress is linked to the runtime turn audit events that were
actually written.
