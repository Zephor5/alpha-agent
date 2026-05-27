# Session Context Handover Compression Plan

## Status

Implemented and verified. Phase 1-6 explicit APIs, runtime orchestration,
verification coverage, and documentation cleanup are complete:
`session_messages` is the durable source stream, `SessionContextAssembler`
projects LLM-visible context from the latest compressed-message boundary,
deterministic context budget primitives are in place, tool replay payload
truncation maintenance is available, callable handover compression appends
`compressed_message` records after a successful provider call, and pre-user plus
tool-loop maintenance are wired into the serialized runtime turn lifecycle.

## Core Direction

Use one durable source of truth for session information. LLM context and
cognition projections both read from that source, but they consume different
message kinds.

Compression is not a separate state table and does not rewrite the whole
session. A successful LLM handover compression appends one special
`compressed_message` source record. Future LLM context assembly consumes the
latest `compressed_message` for the session and skips older source messages
that it replaces. Cognition source mapping ignores `compressed_message` by
default.

Tool payload truncation is a separate, lower-cost maintenance step. It directly
normalizes source tool payloads because large tool inputs/results are often
duplicative and do not need exact long-term replay fidelity.

## Current Foreground Role

`ContextWindow.foreground` is cognition working memory, not LLM context.

It currently exists to support the cognition loop:

- keep a bounded set of recent perceived stimuli per cognition/conversation
  thread;
- recover `Perception` objects from the append-only cognitive event log;
- carry counterpart and situation references for interpretation, judgment, and
  recall;
- preserve explicitly anchored perceptions while rolling older unanchored
  perceptions out of foreground;
- let consolidation move old foreground perceptions into a background reference.

It is not replayable chat history. It does not contain full assistant responses
or complete runtime interaction history. LLM context should be assembled from
source session messages, not from foreground.

## Source Message Model

The source message stream is append-only by ordinal. Append-only here means
message order and identity are stable; tool replay payload truncation below is
the only planned in-place content normalization. The stream should support these
message kinds:

- `user_message`: original user input.
- `assistant_message`: assistant output.
- `tool_message`: runtime/tool output that is part of replayable context.
- `compressed_message`: LLM-produced handover that replaces earlier context for
  LLM replay.

`compressed_message` is a synthetic source record. It is consumed by the LLM
context projection and skipped by cognition projections by default. A later
cognition layer may explicitly read `compressed_message` as evidence for
higher-level facts, but that is a separate extraction rule.

Tool truncation is the only planned in-place mutation of source content. Here
source content means durable replay records for original tool input and output.
Those tool input/output records are JSON structures; truncation deep-walks the
JSON and rewrites only string fields inside those JSON structures.

Suggested source schema:

```sql
CREATE TABLE session_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    llm_role TEXT,
    raw_content TEXT NOT NULL,
    model_content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_result_id TEXT,
    provider_metadata TEXT NOT NULL DEFAULT '{}',
    source_metadata TEXT NOT NULL DEFAULT '{}',
    compression_point_ordinal INTEGER,
    compression_version TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(session_id, ordinal),
    CHECK (ordinal >= 1)
);
```

The previous `conversation_messages` table is directly refactored into this
shape as `session_messages`. The project does not require database
compatibility. The source model must preserve
complete tool replay data (`tool_call_id`, assistant `tool_calls`, and
`tool_result_id`) and keep provider metadata, source metadata, and general
metadata separate.

## Provider Context Budget

Configure max context length per LLM provider/model. Context budget checks must
count:

- LLM-visible messages;
- tool schema tokens;
- expected output reserve;
- a safety margin.

Tool schema tokens matter because tool definitions are sent alongside messages
and consume the same model context window.

Recommended config shape:

```toml
[llm.context]
tool_truncate_threshold_ratio = 0.60
handover_compress_threshold_ratio = 0.90
minimum_remaining_tokens = 10000
tool_string_truncate_chars = 300
expected_output_reserve_tokens = 4096
safety_margin_tokens = 1024

[llm.providers.openai-compatible]
max_context_tokens = 258400

[llm.providers.deepseek]
max_context_tokens = 1000000
```

If per-model overrides become necessary, add them explicitly under the provider
configuration. The numeric values above are deployment defaults, not universal
provider guarantees.

Budget terms:

```text
used_context =
  message_tokens
  + tool_schema_tokens
  + expected_output_reserve_tokens
  + safety_margin_tokens

remaining_context = max_context_tokens - used_context
```

Initial token estimation should be simple and deterministic:

- English-like text counts by word count.
- Chinese text counts by CJK character count.
- Mixed text adds both counts.
- JSON payloads and tool schemas are serialized deterministically and estimated
  with the same text rule.

## Tool Context Truncation

When projected context length exceeds 60% of provider max context, run tool
context truncation before attempting handover compression.

Scope:

- current session only;
- messages after the latest `compressed_message`;
- tool-related source messages that have not been checked for truncation;
- tool input JSON, including assistant tool-call arguments stored in the
  replayable `tool_calls` payload;
- tool output JSON, including tool result content stored in replayable message
  content;
- no truncation of provider metadata, source metadata, or non-replay diagnostic
  metadata.

Algorithm:

1. Find the latest `compressed_message`, if any.
2. Scan later source messages for unchecked tool input/output payloads.
3. Parse tool input/output JSON.
4. Deep-walk objects and arrays.
5. For every string longer than `tool_string_truncate_chars`, replace it with a
   shortened string that preserves the beginning and includes:

```text
<system-reminder>truncated</system-reminder>
```

Example:

```json
{
  "body": "original long text ... <system-reminder>truncated</system-reminder>"
}
```

6. Write the normalized payload back to the source message.
7. Mark the message metadata with:
   - `truncate_checked = true`
   - `original_lengths`

Do not store `truncate_version`, `truncated_paths`, or hashes. The marker in the
content is enough to show that truncation happened, and `original_lengths` is
enough diagnostic metadata. `metadata` may record truncation diagnostics, but
`provider_metadata`, `source_metadata`, and tool-result diagnostic metadata must
not be rewritten as a substitute for replay payload truncation.

If JSON parsing or truncation fails, raise the exception directly. Do not mark
the message as checked, do not silently skip it, and do not continue with a
partially truncated payload.

After truncation, rebuild the projected context and re-estimate tokens.

## Handover Compression Trigger

Run LLM handover compression when either condition is true after any tool
truncation pass:

- projected context length exceeds 90% of provider max context;
- remaining context window is less than `minimum_remaining_tokens`.

Compression should run as early as possible before processing a new user
message. For a new inbound user message:

1. Estimate existing session context plus the pending user message.
2. If the pending user message itself cannot fit even after prior context
   compression, reject it directly.
3. If maintenance is needed, truncate/compress existing source context before
   appending or processing the pending user message.
4. Then append/process the current user message normally.

The pending user message is used for budget planning, but pre-turn compression
does not fold that pending message into the handover because it has not yet
become processed session context.

During a tool loop, compression may be forced if the context crosses the
handover threshold. It must only happen at a safe replay boundary:

```text
assistant/tool request
tool result
request-only user: <system-reminder>compression prompt</system-reminder>
assistant: compressed output
append compressed_message
next LLM call
```

There must be no unresolved tool execution request when the compression prompt
is added to the request. The prompt belongs after the tool result so the
compressed output can include the complete latest task state. The compression
prompt is never persisted as source content.

## Handover Call Shape

The compressor is the same kind of LLM call as the normal runtime call over the
current visible context, not a separate summarization API over a selected
message list. Use the same provider/model, tool definitions, and tool-choice
behavior, and preserve the same prompt prefix; the only prompt difference is the
final transient compression instruction.

Compression call:

```text
current projected context
user: <system-reminder>compression prompt</system-reminder>
```

The compression prompt is transient. It is sent as the last message of the
compression call and is not persisted as conversation content.

The model output is persisted as a new `compressed_message`. When appending that
source record, wrap the returned text in a user-role `<system-reminder>`.

## Compressed Message, Not Summary

The `compressed_message` content should be operational continuity context, not a
short summary. Compression changes the model's local context. The next model
call is a new context holder taking over from the prior context, so the artifact
must preserve operational continuity.

The compressed content must answer:

- What conversation/task is currently in progress?
- What has already been decided or completed?
- What constraints, preferences, and commitments must remain active?
- What is the latest concrete working context needed to continue the active
  task accurately?
- What should the next assistant know before responding?
- What source messages or artifacts support the handover claims?

Recommended content body. Runtime wraps this body in `<system-reminder>` when
persisting the `compressed_message`:

```markdown
## Continuity Contract
- This handover replaces context through compression point ordinal N.
- Continue as the same assistant in the same session.
- Treat claims here as derived from prior source context, not as new user input.

## Active User Intent
- ...

## Current Task State
- ...

## Latest Working Context
- Current step:
- Relevant immediate inputs:
- Known partial results:
- Local assumptions:
- Blockers or risks:
- Next likely action:

## Decisions And Commitments
- ...

## Completed Actions
- ...

## User Preferences And Constraints
- ...

## Open Questions Or Pending Requests
- ...

## Important Supporting Context
- ...

## Do Not Lose
- ...
```

The latest working context should be more detailed than older completed context.
This may include tool-derived facts, but tool calls are just one possible source
of task state and should not be treated as the main axis of the handover.

## Compressed Message Semantics

Append one `compressed_message` when compression succeeds:

```text
session_id
ordinal
kind = compressed_message
llm_role = user
raw_content = <system-reminder> compressed continuity content </system-reminder>
compression_point_ordinal
compression_version
created_at
metadata
```

`compression_point_ordinal` is the last source ordinal covered by the compressed
message. It is a diagnostic and replay boundary. LLM context assembly should
include:

```text
system identity
latest compressed_message as user-role system-reminder context
source messages with ordinal > compressed_message.ordinal
```

Using `compressed_message.ordinal` as the replay boundary prevents the handover
message itself, or source messages already followed by that handover, from being
duplicated in future prompts.

The runtime does not need to record all source message ids. The compressed
message is produced from the full visible context before the compression prompt,
and the durable boundary is the compression point on the appended message.

Latest `compressed_message` by session ordinal wins. Older compressed messages
remain in the source stream for audit but are not selected for LLM context
assembly.

## Append Preconditions

Do not validate the model-produced compressed content. Before appending
`compressed_message`, only enforce source-state preconditions:

- selected `compression_point_ordinal` is recorded on the message;
- compressed content does not treat the compression instruction itself as session
  content;
- compressed content does not include any future or not-yet-persisted user
  message.

If appending fails, do not drop source messages. Fail loudly rather than
silently discarding history.

## Prefix Stability Rules

- The first message remains the only `system` role message.
- Normal user/assistant/tool source messages append by ordinal.
- Tool truncation may rewrite tool input/output payload content in place, but it
  must not change message role, order, ids, or metadata needed for replay.
- Compression appends `compressed_message`; it does not delete covered source
  messages.
- Future LLM prompts use the latest `compressed_message` and source messages
  after that compressed message's own ordinal.
- Dynamic cognition context should not be inserted between stable source
  messages. If cognition-derived context must become LLM-visible, persist it as
  source context or fold it into the next handover.

## Turn Serialization

Turns for the same session must run serially. No two turns for the same
`session_id` may interleave source loading, truncation, compression, source
message appends, main LLM calls, or tool loops. Different sessions may run
concurrently.

## Runtime Pipeline

Introduce a session context assembly service:

```text
SessionContextAssembler
  load_source_messages(session_id)
  find_latest_compressed_message(session_id)
  assemble_projected_context(...)
  estimate_context_tokens(...)
  reject_if_pending_user_too_large(...)
  truncate_tool_context_if_needed(...)
  compress_with_handover_if_needed(...)
  append_compressed_message(...)
  assemble_messages_for_llm(...)
```

Pre-user turn pipeline:

```text
receive pending user message
load source messages
assemble projected context for budget check
reject pending user if it cannot fit
if context > 60%: truncate unchecked tool context
rebuild and re-estimate
if context > 90% or remaining < 10000: append compressed_message through LLM compression
rebuild context
append/process current user message
main LLM call
```

Tool-loop pipeline:

```text
assistant emits tool request
execute tool
append tool result
rebuild and estimate context
if context > 60%: truncate unchecked tool context
rebuild and re-estimate
if context > 90% or remaining < 10000:
  add transient compression prompt to request after tool result
  call LLM for compression
  append compressed_message
  rebuild context
continue next LLM call
```

## Relationship To Cognition

Cognition and LLM context share the same source stream, but detailed cognition
source mapping is intentionally out of scope for this document.

Default cognition source mapping:

- leave `user_message`, `assistant_message`, and `tool_message` available for
  future cognition mapping;
- skip `compressed_message`;
- keep `ContextWindow.foreground` as cognition working memory, not LLM context.

A later cognition layer may explicitly read `compressed_message` as evidence for
higher-level facts, but that extraction must be explicit so synthetic compressed
content does not silently become raw user/assistant fact.

## Testing Plan

Required tests:

- two-turn prompt is append-only with no compression;
- tool replay fields survive the source-message schema refactor;
- provider max context length is loaded and used in estimates;
- token estimates use English word count plus Chinese character count and
  include messages, tool schema tokens, output reserve, and safety margin;
- context over 60% truncates unchecked tool input/output JSON strings longer
  than 300 characters;
- truncation writes back to source tool payloads, marks `truncate_checked`, and
  records `original_lengths`;
- truncation does not modify provider metadata, source metadata, or non-replay
  diagnostic metadata;
- pending user message that is too large is rejected directly;
- context over 90% or remaining window below 10000 triggers handover
  compression;
- pre-user compression appends `compressed_message` before the pending user is
  processed;
- failed pre-user maintenance does not persist the pending user message as a
  processed session message;
- tool-loop compression only happens after a tool result, never while a tool
  request is unresolved;
- compression call adds a transient user role `<system-reminder>` instruction to
  the current visible context;
- compression instruction is not persisted as source content;
- after compression, the next prompt starts with the same system identity and
  latest `compressed_message` until a newer compression occurs;
- LLM context assembly uses source messages after `compressed_message.ordinal`;
- cognition source mapping ignores `compressed_message` by default.

## Implementation Phases

### Phase 1: Source Stream And Turn Boundary

Establish the durable source layer and prevent concurrent same-session mutation
before adding maintenance behavior.

- Refactor the previous transcript table into the source message model with `kind`,
  `llm_role`, `compression_point_ordinal`, `compression_version`, `metadata`,
  and `updated_at`.
- Preserve replay fields: `raw_content`, `model_content`, `tool_call_id`,
  `tool_calls`, `tool_result_id`, `provider_metadata`, and `source_metadata`.
- Add append/read helpers for normal source messages and `compressed_message`.
- Define latest `compressed_message` lookup as max session ordinal where
  `kind = compressed_message`; do not add status or active-state machinery.
- Add session-level turn serialization so one `session_id` cannot interleave
  source loading, maintenance, appends, LLM calls, or tool loops.
- Keep cognition handling minimal: `compressed_message` is ignored by default;
  detailed source-to-cognition mapping remains out of scope.

### Phase 2: Context Projection And Budgeting

Build one context assembly path that every LLM-facing caller will use.

- Implement `SessionContextAssembler` over the source stream.
- Assemble LLM context as: stable system identity, latest `compressed_message`
  if present, then source messages after that compressed message's ordinal.
- Add provider/model `max_context_tokens` and context threshold/reserve
  configuration.
- Implement deterministic token estimation: English word count plus Chinese CJK
  character count, including deterministic JSON serialization for tool schemas
  and payloads.
- Estimate existing context plus pending user message before appending the
  pending user message.
- Reject a pending user message that cannot fit after prior context maintenance.
- Do not reduce ordinary transcript replay outside explicit compression.

### Phase 3: Tool Payload Maintenance

Add the low-cost maintenance path that can reduce context without invoking the
model.

- Add a narrowly scoped update helper for replay payload truncation.
- Scan only source messages after the latest `compressed_message`.
- Parse tool input/output records as JSON and deep-walk objects and arrays.
- Truncate string fields longer than `tool_string_truncate_chars` with the
  `<system-reminder>truncated</system-reminder>` marker.
- Persist normalized JSON back to the source tool input/output record.
- Mark `truncate_checked` and record `original_lengths` in general metadata.
- Leave provider metadata, source metadata, and non-replay diagnostic metadata
  unchanged.
- Let JSON parse or truncation failures raise directly.

### Phase 4: Compression Call And Source Append

Add LLM compression as a normal model call plus a new source message append.

- Build the compression request from the current projected context plus one
  transient final user-role `<system-reminder>` compression instruction.
- Use the same provider/model, tool definitions, tool-choice behavior, and prompt
  prefix as a normal runtime call.
- Do not persist the compression instruction.
- Wrap the model output in `<system-reminder>` and append it as
  `kind = compressed_message`, `llm_role = user`.
- Set `compression_point_ordinal` to the last source ordinal covered before the
  transient compression prompt.
- Do not validate or repair the model-produced compressed content.
- Persist compression started/completed/failed traces for diagnostics only.

### Phase 5: Runtime Orchestration

Wire the maintenance paths into the actual turn lifecycle.

- Run pre-user maintenance before appending or processing a pending user message.
- If pre-user maintenance fails, do not persist the pending user as a processed
  session message.
- After tool execution, append the tool result, rebuild context, then run
  truncation and compression only if thresholds require it.
- In tool loops, add the transient compression prompt only after a complete tool
  result and never while a tool request is unresolved.
- Rebuild projected context after truncation and after compression before the
  next LLM call.
- Route main runtime LLM calls and debug prompt rendering through the same
  `SessionContextAssembler`.

### Phase 6: Full Verification And Documentation Cleanup

Verify the completed path as an integrated feature rather than treating each
phase as an independently shippable endpoint.

- Cover the required tests above across source model, projection, budgeting,
  truncation, compression, pre-user flow, tool-loop flow, debug prompt, and
  cognition skip behavior.
- Update configuration docs and examples for the final context budget settings.
- Remove or rewrite docs that describe count-based tail selection as normal LLM
  context behavior.
- Confirm no source messages are deleted by compression and no compression
  instruction is persisted as source content.

Suggested verification checkpoints:

- After Phases 1-2: source stream, latest compressed-message lookup, turn
  serialization, context assembly, and budgeting can be tested together.
- After Phases 3-4: tool truncation and compression append semantics can be
  tested together without full runtime orchestration.
- After Phases 5-6: run end-to-end pre-user and tool-loop compression tests plus
  documentation cleanup checks.
