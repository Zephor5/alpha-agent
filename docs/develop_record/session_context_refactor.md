# Session Context Refactor Plan

## Status

Draft for implementation.

## Core Problem

Alpha currently treats `working_memory` as the recent session context. That makes
multi-turn context unstable because it is priority/limit based and can be
pruned independently of the real conversation. The desired model is different:

- Original conversation messages are append-only facts.
- LLM-visible session context is a stable projection of those original messages.
- Context compression is the only normal mechanism that reduces prior context.
- Current user input remains the final raw `user` message in the prompt.
- Retrieved long-term memory remains query-dependent background, not session
  transcript.

So the root change is not to tune `working_memory_limit`; it is to separate
original transcript, session context, and per-turn retrieval.

## Hermes Reference

Hermes has useful implementation patterns in:

- `hermes-agent/agent/conversation_loop.py`
- `hermes-agent/agent/context_engine.py`
- `hermes-agent/agent/context_compressor.py`
- `hermes-agent/agent/conversation_compression.py`
- `hermes-agent/hermes_state.py`
- `hermes-agent/gateway/session.py`
- `hermes-agent/gateway/run.py`

What Alpha should borrow:

- Keep durable transcript, runtime API view, and stable system prompt as separate
  phases.
- Build model requests from a copy of the transcript, not by mutating persisted
  messages.
- Compress middle history only; preserve the latest user request and a recent
  tail.
- Treat assistant tool calls and tool results as an inseparable sequence.
- Include system prompt and tool schemas in token estimates.
- Surface compression failures explicitly.

What Alpha should improve instead of copying:

- Hermes does not fully separate raw inbound content from model-enriched content.
  Alpha should make `raw_content` and `model_content` first-class.
- Hermes has dual SQLite/JSONL transcript truth. Alpha should keep one database
  source of truth.
- Hermes can rewrite/split compressed sessions. Alpha should keep the original
  transcript untouched and store compression as a session-context projection.

## Target Architecture

### 1. Original Message Layer

Create a first-class conversation message store separate from runtime events.
This replaces using `working_memory` as a session transcript.

Required shape:

- `session_id`
- monotonic `ordinal`
- `role`: `user`, `assistant`, or `tool`
- `raw_content`: original inbound or generated content, unexpanded where
  applicable
- `model_content`: content used for model replay when it differs from raw input
- tool-call replay fields: tool call id, assistant tool calls, tool result id
- provider metadata needed for replay/debugging
- source metadata such as gateway platform/message ids
- timestamps and general metadata

The current generic `events` table should not remain the source of truth for
conversation content. If retained, it should be narrowed into diagnostic traces
only.

### 2. Session Context Layer

Add a session context projection built from original messages:

```text
active session context =
  optional compressed summary of earlier messages
  + uncompressed original messages after compressed_until_ordinal
```

Store one active context state per session:

- `session_id`
- `compressed_until_ordinal`
- `summary`
- `summary_source_message_ids`
- `compression_version`
- `created_at`
- `updated_at`
- metadata with token counts and compression reason

The original message layer remains complete even after compression. Rebuilding
or re-compressing context must not delete or rewrite original messages.

### 3. Event Model Simplification

The current `events` model is too broad: it stores raw conversation content,
runtime lifecycle records, tool activity, memory retrieval markers, and failure
diagnostics in the same abstraction. That creates redundant data once
conversation messages and session context become first-class.

Target model:

- `conversation_messages`: source of truth for user, assistant, and tool
  transcript content.
- `session_context_states`: active compressed context projection for a session.
- `runtime_traces`: optional diagnostic records for LLM/tool/context operations
  that are useful for debugging but not needed to rebuild conversation state.
- `memory_access_log`: existing retrieval/access diagnostics for long-term
  memory.

Persist only traces that explain behavior or support debugging:

- LLM request/response summaries, trace ids, provider names, token counts, and
  retry counts.
- Tool execution results and failures that affected the final turn.
- Context compression started/completed/skipped/failed records.
- Memory extraction and persistence results.
- Gateway dedup/source linkage when it affects delivery or replay.

Do not persist generic lifecycle records by default:

- `turn.started`
- routine `turn.completed` when conversation messages already capture the
  result
- routine `memory.retrieved` when `memory_access_log` already records retrieved
  ids and scores
- duplicated user/assistant/tool content that already exists in
  `conversation_messages`

The implementation can either replace `events` with `runtime_traces`, or keep
the table name temporarily while narrowing its writes and schema. Because this
project does not require database compatibility, prefer the direct rename/schema
refactor if it makes the resulting model cleaner.

### 4. Prompt Assembly

The normal prompt shape should be:

```text
system:
  stable Alpha identity and behavior rules

user:
  <system-reminder>
  retrieved long-term memory, runtime reminders, and other non-transcript
  background for this turn
  </system-reminder>

prior conversation:
  optional compressed session summary as a reference-only user message
  prior uncompressed user/assistant/tool messages in order

user:
  current raw user message
```

Rules:

- The first message is the only `system` role message.
- The current user message is always last and unwrapped.
- Session context is not query-dependent.
- Long-term semantic/episodic/procedural retrieval is query-dependent and belongs
  in `<system-reminder>`, not in the transcript.
- Tool-loop finalization remains a `user` role `<system-reminder>`.

### 5. Context Compression

Replace `working_memory_limit` with context-budget configuration:

- `context.max_prompt_tokens`
- `context.compression_threshold_ratio`
- `context.recent_tail_messages`
- `context.min_summary_tokens`
- `context.max_summary_tokens`

Compression trigger:

- Estimate prompt tokens before the LLM call using system prompt, reminders,
  session context, current user message, and tool schemas.
- If the estimate exceeds the threshold, compress the oldest middle context.
- Never compress the current user message.
- Preserve a recent tail of original messages.
- Preserve tool-call/result integrity.

Compression output should be structured:

```text
## Compressed Session Context

### Active Task
...

### Decisions
...

### Completed Actions
...

### Current State
...

### Pending User Requests
...

### Relevant Files Or Artifacts
...
```

Repeated compression updates or replaces the active summary and advances
`compressed_until_ordinal`; it never deletes the original transcript.

## Implementation Plan

### Phase 1: Rename The Model Boundary

- Introduce conversation-message domain models and store methods.
- Stop writing user/assistant/tool conversational content only as generic
  runtime events.
- Replace broad runtime events with narrow diagnostic traces. Do not persist
  routine turn lifecycle records unless they carry information not represented
  elsewhere.
- Remove `WorkingMemoryManager` from the agent turn path.
- Remove `working_memory_limit` from config, CLI output, docs, and tests.
- Remove `working_memory` prompt section and replace it with session context.

### Phase 2: Build Stable Session Context

- Add a `SessionContextManager` that loads:
  - active compressed context state for the session
  - uncompressed conversation messages after `compressed_until_ordinal`
- Build prompt messages from the session context before appending the current
  raw user message.
- Persist the current user message with an ordinal, but exclude it from prior
  context by querying messages with ordinal lower than the current message.
- Persist assistant responses and tool results into the original message layer
  after they occur.
- Keep debug prompt output showing both session context and retrieved memory.

### Phase 3: Add Compression Engine

- Add a `ContextCompressor` interface with:
  - `should_compress(context, budget) -> bool`
  - `compress(messages, previous_summary, focus) -> CompressionResult`
- Implement the first compressor with the configured LLM provider where
  available, and a deterministic fallback for mock tests.
- Compress only messages older than the preserved recent tail.
- Include token accounting in compression metadata.
- Emit diagnostic traces for compression started, completed, skipped, and
  failed.

### Phase 4: Clean Up Long-Term Memory Retrieval

- Keep semantic, episodic, and procedural retrieval independent from session
  context.
- Rename `RetrievedContext.working_memory` to session-context or remove it from
  retrieval entirely.
- Ensure memory extraction reads original conversation messages, not compressed
  summaries.
- Keep procedural memory body gating based on current user request relevance.

### Phase 5: Remove Redundant Event Writes

- Replace `_emit_turn_started`, `_emit_turn_completed`, and routine
  `memory.retrieved` writes with direct debug metadata or targeted trace records.
- Persist user, assistant, and tool content only through `conversation_messages`.
- Keep failure traces for observable errors, but store the final answer and tool
  replay in the conversation-message layer.
- Update CLI/debug views to read conversation history from
  `conversation_messages` and operational diagnostics from `runtime_traces`.

## Testing Plan

Core tests:

- A two-turn conversation sends the first turn as prior session context and the
  second raw user message as the final prompt message.
- No normal prompt has more than one `system` message.
- Retrieved memory appears inside `<system-reminder>` and is not persisted as
  original transcript.
- Removing `working_memory_limit` removes all priority/limit pruning behavior
  from the turn path.
- Original conversation messages remain complete after context compression.
- Compression replaces older prompt context with a summary while preserving the
  recent tail.
- Tool-call assistant messages and matching tool results are not split by
  compression.
- Debug prompt displays original session context, compressed summary if present,
  retrieved memory ids, and token estimates.
- Routine user/assistant/tool content is not duplicated into diagnostic traces.
- Diagnostic traces retain enough information to inspect LLM calls, tool
  failures, compression, and memory persistence.

Regression tests:

- Current user message is never included in compressed prior context.
- Context compression failure does not silently drop transcript.
- Mock provider remains deterministic.
- OpenAI-compatible and Codex providers receive replayable tool messages in the
  same order as before.

## Direct Refactor Notes

Project rules say not to preserve compatibility with existing database data.
Implement this as a direct schema/runtime refactor:

- Drop `working_memory` as a prompt-context mechanism.
- Remove `working_memory_limit` rather than preserving it as a deprecated alias.
- Replace broad `events` usage with explicit conversation-message and
  diagnostic-trace storage.
- Update tests and docs to the new model in one pass.
- Do not add compatibility shims for old working-memory rows or old generic
  event rows.
