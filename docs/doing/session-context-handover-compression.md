# Session Context Handover Compression Plan

## Status

Draft for implementation.

## Core Direction

Use one durable source of truth for session information. LLM context and
cognition projections both read from that source, but they consume different
message kinds.

Compression is not a separate state table and does not rewrite the whole
session. A successful LLM handover compression appends one special
`compressed_message` source record. Future LLM context assembly consumes the
latest valid `compressed_message` and skips older source messages that it
replaces. Cognition source mapping ignores `compressed_message` by default.

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

The source message stream is append-only by ordinal. It should support these
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

Tool truncation is the only planned in-place mutation of source content. It is
allowed only for tool input/output payloads and must mark the resulting content
as truncated.

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
    compression_point_ordinal INTEGER,
    compression_version TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(session_id, ordinal)
);
```

Existing `conversation_messages` can be directly refactored into this shape. The
project does not require database compatibility.

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

Budget terms:

```text
used_context =
  message_tokens
  + tool_schema_tokens
  + expected_output_reserve_tokens
  + safety_margin_tokens

remaining_context = max_context_tokens - used_context
```

## Tool Context Truncation

When projected context length exceeds 60% of provider max context, run tool
context truncation before attempting handover compression.

Scope:

- current session only;
- messages after the latest active `compressed_message`;
- tool-related source messages that have not been checked for truncation;
- tool input JSON, including assistant tool-call arguments;
- tool output JSON, including tool result content;
- no truncation of tool metadata.

Algorithm:

1. Find the latest active `compressed_message`, if any.
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
enough diagnostic metadata.

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
user: <system-reminder>compression prompt</system-reminder>
assistant: handover output
append compressed_message
next LLM call
```

There must be no unresolved tool execution request when the compression prompt
is appended. The prompt belongs after the tool result so the handover can include
the complete latest task state.

## Handover Call Shape

The compressor is a normal LLM call over the current visible context, not a
separate summarization API over a selected message list.

Compression call:

```text
current projected context
user: <system-reminder>compression prompt</system-reminder>
```

The compression prompt is transient. It is sent as the last message of the
compression call and is not persisted as conversation content.

The model output is persisted as a new `compressed_message`.

## Handover, Not Summary

The `compressed_message` content should be a handover document, not a short
summary. Compression changes the model's local context. The next model call is a
new context holder taking over from the prior context, so the artifact must
preserve operational continuity.

A handover must answer:

- What conversation/task is currently in progress?
- What has already been decided or completed?
- What constraints, preferences, and commitments must remain active?
- What is the latest concrete working context needed to continue the active
  task accurately?
- What should the next assistant know before responding?
- What source messages or artifacts support the handover claims?

Recommended content structure:

```markdown
<session-handover version="1">

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

</session-handover>
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
raw_content = handover content
compression_point_ordinal
compression_version
status = active
created_at
metadata
```

`compression_point_ordinal` is the last source ordinal covered by the handover.
It is a diagnostic and validation boundary. LLM context assembly should include:

```text
system identity
latest active compressed_message as user-role handover context
source messages with ordinal > compressed_message.ordinal
```

Using `compressed_message.ordinal` as the replay boundary prevents the handover
message itself, or source messages already followed by that handover, from being
duplicated in future prompts.

The runtime does not need to record all source message ids. The handover is a
compression of the full visible context before the compression prompt, and the
durable boundary is the compression point on the appended message.

Latest valid `compressed_message` wins. Older compressed messages remain in the
source stream for audit but are not selected for LLM context assembly.

## Validation

Before appending `compressed_message`, validate:

- output contains all required handover sections;
- selected `compression_point_ordinal` is recorded on the message;
- handover does not treat the compression instruction itself as session content;
- handover does not include any future or not-yet-persisted user message;
- handover stays within configured token/character limits;
- handover is non-empty for active tasks and records uncertainty when the source
  is ambiguous.

If validation fails, do not drop source messages. Retry with a repair prompt if
there is room. If the context still cannot fit, fail loudly with a compression
error trace rather than silently discarding history.

## Prefix Stability Rules

- The first message remains the only `system` role message.
- Normal user/assistant/tool source messages append by ordinal.
- Tool truncation may rewrite tool input/output payload content in place, but it
  must not change message role, order, ids, or metadata needed for replay.
- Compression appends `compressed_message`; it does not delete covered source
  messages.
- Future LLM prompts use the latest active `compressed_message` and source
  messages after that compressed message's own ordinal.
- Dynamic cognition context should not be inserted between stable source
  messages. If cognition-derived context must become LLM-visible, persist it as
  source context or fold it into the next handover.
- `recent_tail_messages` must not truncate by itself.

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
if context > 90% or remaining < 10000: append compressed_message through LLM handover
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
  append compression prompt after tool result
  call LLM for handover
  append compressed_message
  rebuild context
continue next LLM call
```

## Relationship To Cognition

Cognition and LLM context share the same source stream, but they do not consume
it identically.

Default cognition source mapping:

- consume `user_message`, `assistant_message`, and relevant runtime messages as
  cognition inputs;
- skip `compressed_message`;
- keep `ContextWindow.foreground` as cognition working memory, not LLM context.

Future cognition integration:

- foreground can influence whether compression is useful or urgent;
- foreground anchors can become handover preservation hints;
- a later cognition layer can explicitly read `compressed_message` as evidence
  for higher-level facts;
- that extraction must be explicit so synthetic handover content does not
  silently become raw user/assistant fact.

## Testing Plan

Required tests:

- two-turn prompt is append-only with no compression;
- provider max context length is loaded and used in estimates;
- token estimates include messages, tool schema tokens, output reserve, and
  safety margin;
- context over 60% truncates unchecked tool input/output JSON strings longer
  than 300 characters;
- truncation writes back to source tool payloads, marks `truncate_checked`, and
  records `original_lengths`;
- truncation does not modify tool metadata;
- pending user message that is too large is rejected directly;
- context over 90% or remaining window below 10000 triggers handover
  compression;
- pre-user compression appends `compressed_message` before the pending user is
  processed;
- tool-loop compression only happens after a tool result, never while a tool
  request is unresolved;
- compression call appends a user role `<system-reminder>` instruction to the
  current visible context;
- compression instruction is not persisted as source content;
- after compression, the next prompt starts with the same system identity and
  latest `compressed_message` until a newer compression occurs;
- LLM context assembly uses source messages after `compressed_message.ordinal`;
- cognition source mapping ignores `compressed_message` by default.

## Implementation Phases

### Phase 1: Source Message Model

- Refactor `conversation_messages` into a source message model with `kind`,
  `llm_role`, `compression_point_ordinal`, `compression_version`, `status`,
  `metadata`, and `updated_at`.
- Add append/read helpers for `compressed_message`.
- Make cognition source mapping ignore `compressed_message` by default.

### Phase 2: Context Budgeting

- Add provider/model `max_context_tokens`.
- Add context threshold/reserve configuration.
- Implement context token estimation including messages, tool schemas, output
  reserve, and safety margin.
- Reject pending user messages that cannot fit.

### Phase 3: Tool Context Truncation

- Implement JSON deep truncation for unchecked tool input/output payloads.
- Truncate strings longer than 300 characters with the
  `<system-reminder>truncated</system-reminder>` marker.
- Persist normalized payloads in source messages.
- Mark `truncate_checked` and record `original_lengths`.

### Phase 4: LLM Handover Compression

- Implement compression as a normal LLM call over current context plus a final
  user role `<system-reminder>` compression instruction.
- Validate handover output.
- Append successful handovers as `compressed_message`.
- Persist compression started/completed/failed traces for diagnostics only.

### Phase 5: Runtime Integration

- Run truncate/compress preflight before processing current user messages where
  possible.
- Run forced truncate/compress inside tool loops only after tool results.
- Ensure main LLM calls and debug prompt use the same context projection.
