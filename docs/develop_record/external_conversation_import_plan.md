# External Conversation Import Execution Plan

## Status

Phases 1-7 implemented. Phase 8 documentation is complete. Final full-suite
validation and end-to-end smoke import remain open release gates.

## Date

2026-06-10

## Source Of This Plan

This plan records the decisions from the product and architecture alignment for
importing external LLM conversation records into Alpha Agent. It is intentionally
self-contained and does not rely on other todo or historical documents.

## Core Goal

Alpha Agent is a personal dedicated agent. External LLM conversations are a
valuable cognition source because they contain the user's past self-descriptions,
preferences, constraints, projects, decision patterns, and long-running goals.

The import feature should make those external user conversations available to
the existing cognition pipeline so Alpha can build a more complete long-term
understanding of the user.

Success for the first version means:

- A valid normalized JSON file can be imported through the daemon boundary.
- Imported conversations are retained as hidden, session-shaped source material.
- Imported sessions are invisible to normal chat and cannot be continued.
- Existing cognition processing can consume imported session messages.
- Import status can report import counts and extraction progress.
- The feature does not invent a separate memory system.

Success does not require perfect immediate profile quality. User profile quality
continues to depend on the existing extraction, consolidation, and summary
pipeline.

## Product Principles

### Unified User Identity

All imported `user` messages represent the current owner of this personal agent.
The import source does not create separate user identities or source-specific
profiles.

Accepted principles:

- The imported user is "me" by default.
- Multiple providers merge into the same user cognition.
- Source provider does not affect memory authority or weight.
- Time and memory type can affect which cognition is current.
- External assistant output is not user self-description unless the user adopts
  or responds to it in a way that makes it evidence.

### Two-Layer Model

The feature has two conceptual layers:

- Raw record layer: preserve external conversation messages as hidden source
  material.
- Cognition digestion layer: existing background cognition extracts and
  consolidates long-term memories from that source material.

The raw record layer is preserved for future deliberate retrieval, but first
version does not build raw message search, show, or list commands.

### No Session Scope Memories

Imported conversations are evidence containers. They are not useful long-term
memory scopes.

Imported conversation extraction should not intentionally produce `SESSION`
scope memory. Durable outputs should naturally land in user, self, counterpart,
or project cognition through the existing pipeline.

First-version import extraction must adjust the extraction prompt and validation
context for import sessions:

- Do not allow `SESSION` scope for imported conversation extraction.
- Do not include the import session itself as an allowed `about` reference.
- Treat imported assistant output as evidence only when the surrounding user
  response adopts, corrects, or otherwise makes that assistant output evidence
  about the user.
- Keep ordinary session extraction semantics unchanged.

### Time Model

Imported messages always have original message timestamps.

Rules:

- Message `created_at` is required and must include an explicit timezone offset
  or `Z`. Naive timestamps are invalid.
- Timestamp comparisons use parsed UTC instants, not raw strings.
- Message `created_at` must be strictly increasing within each external
  conversation across all imported roles: `system`, `user`, `assistant`, and
  `tool`.
- Use external message `created_at` as the persisted session message time. It
  may be normalized to UTC for storage, but it must represent the same instant.
- Do not replace original message time with import time.
- Store import time separately in import batch/message records.
- Conversation-level `created_at` and `updated_at` are metadata/status fields.
  They do not participate in cognition source-time decisions.
- Import does not generate Alpha runtime `session_time` reminders. Imported
  `system` messages are external source messages, not Alpha runtime reminders.
- Cognition uses source message time as approximate evidence time when
  extracting, consolidating, and summarizing memories. `held_since` and
  `observed_at` remain Alpha processing/holding time and must not be treated as
  evidence time.
- The first version uses a single approximate `source_time` for each extraction
  source window, not a per-draft or per-message time range. The approximate
  `source_time` is the latest `created_at` among the source messages selected
  for that window.
- `source_time` is for cognition prompting and recency decisions, not for
  audit-grade proof that a belief came from one exact message or second.
- More recent user expressions can supersede older ones, especially for volatile
  preferences, active projects, tools, and short-term plans.
- Stable long-term values, quality standards, working style, and repeated
  preferences can accumulate across time.

## Explicit Non-Goals For First Version

- No raw conversation search.
- No raw conversation `show` or `list` command.
- No platform-specific export parser.
- No attachment, image, file, OCR, or multimodal support.
- No `--process-now`.
- No extra "update profile now" command.
- No cognitive event for "conversation imported".
- No runtime trace for import.
- No compressed handover message generation during import.
- No import-specific memory store.
- No source-based memory weighting.
- No migration runner or compatibility migration chain.
- No CLI direct database writes.
- No automatic daemon startup.
- No CLI JSON output mode in first version.

## Normalized Import Format

First version accepts one normalized JSON format. Platform exports must be
converted into this format before import.

### Top-Level Shape

```json
{
  "source_provider": "chatgpt",
  "timezone": "Asia/Shanghai",
  "conversations": [
    {
      "external_conversation_id": "conv_001",
      "title": "Alpha Agent design discussion",
      "created_at": "2026-01-01T10:00:00Z",
      "updated_at": "2026-01-01T11:00:00Z",
      "messages": [
        {
          "external_message_id": "msg_001",
          "role": "user",
          "content": "I prefer direct feedback.",
          "created_at": "2026-01-01T10:01:00Z"
        }
      ],
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

### Required Fields

Top level:

- `source_provider`: required non-empty string. Not restricted to a fixed enum.
- `conversations`: required non-empty array.

Conversation:

- `external_conversation_id`: required non-empty string.
- `messages`: required non-empty array.

Message:

- `external_message_id`: required non-empty string.
- `role`: required; one of `system`, `user`, `assistant`, `tool`.
- `created_at`: required timezone-aware timestamp.
- `content`: required non-empty string except for assistant messages with
  non-empty `tool_calls`.

Tool message:

- `tool_call_id`: required non-empty string.
- `content`: required non-empty string.

Assistant message with tool calls:

- `tool_calls`: allowed only on `assistant` messages.
- `content` may be empty or omitted when `tool_calls` is non-empty.
- Each tool call must include an `id` that can be matched by later `tool`
  messages in the same conversation.
- The matched `tool` message must appear later in file order and therefore have
  a strictly later `created_at`.

### Optional Fields

Top level:

- `metadata`: optional JSON object.
- `timezone`: optional IANA timezone or fixed UTC offset. When absent, each
  hidden import session uses the offset from that conversation's first message
  `created_at`; `Z` is treated as `+00:00`.

Conversation:

- `title`: optional string.
- `created_at`: optional timezone-aware timestamp.
- `updated_at`: optional timezone-aware timestamp.
- `metadata`: optional JSON object.
- Per-conversation `timezone` is not supported in the first version.

Message:

- `metadata`: optional JSON object.

### Rejected Fields And Cases

First version rejects:

- Missing or empty message content, except assistant tool-call messages.
- Roles outside `system`, `user`, `assistant`, `tool`.
- Naive timestamps without timezone/offset.
- Message timestamps that are not strictly increasing by UTC instant within one
  conversation.
- `reasoning_content`.
- Attachments or multimodal fields.
- `tool_call_id` on non-tool messages.
- `tool_calls` on non-assistant messages.
- Tool results without a matching assistant tool call.
- Multiple providers in one file.
- Files larger than the configured first-version limit.

### Size Limit

First version uses a 50 MB payload size limit.

The limit is enforced at the earliest practical first-version boundary:

- CLI checks file size before sending content through IPC.
- Daemon/service may defensively reject an oversized received payload, but first
  version does not require the daemon to solve raw IPC framing limits before JSON
  parsing.

Future streaming or chunked import can relax this, but first version should keep
the boundary simple.

## Import Identity And Idempotency

### Provider

`source_provider` is a stable source namespace such as `chatgpt`, `claude`,
`gemini`, `cursor`, or `manual`.

It is not an enum because the normalized contract should allow future providers
without schema churn.

### Conversation Identity

The tuple below uniquely identifies an external conversation:

```text
source_provider + external_conversation_id
```

This tuple maps to one hidden internal session.

The internal `session_id` should be generated using the existing session id
generator, not by deterministic hashing. Stability comes from the import mapping
table, not from predictable session ids.

### Message Identity

The tuple below uniquely identifies an external message:

```text
source_provider + external_conversation_id + external_message_id
```

Rules:

- If the tuple already exists, skip it as deduped.
- Dedup is based only on external message identity. Do not compare existing
  content, timestamps, tool fields, or metadata for conflicts in first version.
- Deduped messages do not update existing `session_messages` or
  `imported_messages`.
- Do not overwrite existing messages.
- If the same conversation contains new message ids, append them to the same
  hidden import session.
- Message append order follows file order.
- Do not sort by timestamp. The normalized payload must already be in intended
  transcript order.
- For a new conversation, all messages must be strictly increasing by
  `created_at` UTC instant in file order.
- For an existing conversation, plan dedup first. Deduped old messages are
  ignored for append-time validation. The first newly inserted message must be
  strictly later than the latest already imported message in that conversation,
  and all newly inserted messages must then be strictly increasing in file
  order.
- Do not support middle insertion or ordinal rewrites in first version.

### Batch Identity

`import_batch_id` is generated by the service for each import attempt.

Rules:

- The input file does not contain `import_batch_id`.
- Re-importing the exact same file still creates a new batch attempt.
- A duplicate import can result in `new_messages=0` and `deduped_messages=N`.
- `--dry-run` does not create a durable batch.
- Batch identity is for status, statistics, troubleshooting, and grouping an
  import attempt. It is not part of memory weighting.

## Persistence Model

The persistence model must treat import as a first-class source boundary while
reusing session-shaped source messages for cognition.

### Tables

The target model needs these concepts:

- `import_batches`: one import attempt.
- `imported_conversations`: one external conversation mapped to one hidden
  internal session.
- `imported_messages`: one external message identity mapped to one internal
  `session_messages` row.

This does not duplicate message content.

Content ownership:

- `session_messages.raw_content` stores normalized message text.
- `imported_messages` stores external identity, mapping, batch linkage, role,
  timestamps, and dedup/status metadata.
- Optional metadata is stored as JSON where needed.

### Suggested Table Responsibilities

`import_batches`:

- `id`
- `source_provider`
- `input_name`
- `payload_digest`
- `status`
- aggregate counts
- `dry_run` should not be persisted as a normal successful batch
- `created_at`
- `updated_at`
- error summary, if failed
- metadata JSON

`imported_conversations`:

- `id`
- `source_provider`
- `external_conversation_id`
- `session_id`
- `title`
- optional external conversation timestamps
- first import batch id
- latest import batch id
- created/imported timestamps
- metadata JSON
- unique key on `source_provider, external_conversation_id`
- unique key on `session_id`

`imported_messages`:

- `id`
- `source_provider`
- `external_conversation_id`
- `external_message_id`
- `imported_conversation_id`
- `session_message_id`
- `import_batch_id`
- normalized `role`
- external `created_at`
- `imported_at`
- metadata JSON
- unique key on `source_provider, external_conversation_id, external_message_id`
- unique key on `session_message_id`

`imported_messages` records inserted external messages. Deduped messages do not
create new `imported_messages` or `session_messages` rows; their counts are
recorded on the import batch summary. If verbose status needs per-conversation
dedup details for a past batch, store those attempt-level counts in
`import_batches.metadata` or an equivalent batch-result structure.

External message timestamps may be normalized to UTC in `session_messages` and
`imported_messages`, as long as the persisted value represents the same instant.
If the original timestamp literal is useful for troubleshooting, keep it in
metadata; do not use it for sorting or cognition decisions.

### Session Message Changes

Persistent session messages need to support external system messages.

Target changes:

- Add `system` to persistent `llm_role`.
- Add a persistent system source kind, e.g. `system_message`, if needed by the
  message shape model.
- `role=system` imports should persist as system source messages.
- Existing Alpha runtime system prompt remains dynamically generated and is not
  written to `session_messages`.
- Ordinary Alpha chat should not start persisting system prompts.
- Import should not synthesize `system_reminder` messages, including
  `session_time` reminders.

### Hidden Import Sessions

Each imported conversation maps to one hidden import session.

Rules:

- Hidden import sessions are source material for cognition.
- They are not user-visible chat sessions.
- They cannot be continued with `ask`, `chat`, daemon turn requests, gateway
  turns, or future ordinary API chat calls.
- Import session detection should use the import conversation mapping table.
- Do not infer import sessions from individual message metadata.
- If a session is both a gateway session and an import session, treat it as a
  data error and reject ordinary chat use.
- New hidden import session metadata should use external history time:
  `sessions.created_at` is the first imported message time and
  `sessions.updated_at` is the latest imported message time.
- When appending new messages to an existing imported conversation,
  `sessions.created_at` remains unchanged and `sessions.updated_at` advances to
  the newest inserted message time.
- Hidden import session `timezone` is the optional top-level `timezone` when
  provided. Otherwise, derive a fixed offset from the conversation's first
  message timestamp; `Z` becomes `+00:00`.
- Do not validate message timestamp offsets against the chosen session
  timezone. Message offsets define instants; session timezone only controls
  prompt/display rendering.

### Main User Binding

Every imported conversation session should be bound to the main user
counterpart.

Rules:

- Do not infer a new counterpart from external provider account ids.
- No core `external_user_id` field is required.
- If external user account information is useful later, keep it in metadata.

## Daemon And Service Boundary

Import is a daemon-owned service capability, not CLI-local database work.

### Core Service

Introduce a daemon-side application service, conceptually:

```text
ConversationImportService
```

Responsibilities:

- Parse JSON.
- Validate normalized import contract.
- Validate timezone-aware message timestamps and strict conversation ordering.
- Plan writes and dedup counts.
- Reject unsupported historical middle insertion during write planning.
- Execute dry-run without writing.
- Execute real import in transactional writes.
- Create or reuse hidden import sessions.
- Bind imported sessions to the main user counterpart.
- Write import mapping records.
- Return structured summaries.
- Return structured validation errors with paths.
- Report batch status.

The service should depend on store-level persistence APIs, not on CLI rendering.

### CLI Role

The CLI is a client.

For `alpha cognition import conversations <file>`:

- Read file content.
- Check file size before IPC.
- Send content to daemon.
- Send `input_name`, not a persistent absolute path.
- Render the returned summary.

For `--dry-run`:

- Send the same payload with `dry_run=true`.
- Render validation and planned counts.
- Do not create durable import batch records.

For `alpha cognition import status <batch_id>`:

- Send status request to daemon.
- Render aggregate status by default.
- Render conversation-level details only with `--verbose`.

### IPC Contract

Add request types rather than overloading `ask` or `chat_turn`.

Conceptual requests:

```json
{
  "type": "conversation_import",
  "input_name": "chatgpt-export.json",
  "payload_json": "{...}",
  "dry_run": false
}
```

```json
{
  "type": "conversation_import_status",
  "batch_id": "import_...",
  "verbose": false
}
```

Responses should be structured. CLI decides how to render them.

Validation errors should include paths:

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Invalid conversation import payload.",
  "details": [
    {
      "path": "conversations[3].messages[12].created_at",
      "message": "created_at must include a timezone and be strictly later than the previous inserted message"
    }
  ]
}
```

### Daemon Lifecycle

The import command should follow existing daemon-owned command behavior.

Rules:

- Do not auto-start the daemon.
- If daemon is unavailable, report that the daemon must be started.
- Do not fall back to local CLI database writes.

### Concurrency

Import is not an agent turn.

Rules:

- Do not use normal per-session turn guard for import.
- Do serialize import writes or protect them with service-level locking plus
  SQLite transactions.
- Do not parse and validate inside a long write transaction.
- Normal chat should not be globally blocked by import parsing/validation.
- Short SQLite write-lock contention during final write is acceptable.

## Cognition Integration

### Reuse Existing Pipeline

Imported conversations should enter cognition through the same broad source
path as other session-shaped source messages.

Rules:

- Do not create a new memory system.
- Do not create import-specific cognitive event kinds.
- Do not write runtime traces for import.
- Do not generate `compressed_message` rows during import.
- Do not generate `system_reminder` rows during import.
- Do not add a first-version `--process-now` path.
- Let daemon/background cognition process imported messages after import.

### Extraction Prompt For Import Sessions

Import sessions require a narrow extraction specialization.

Current background extraction for inactive sessions prepends Alpha's runtime
system prompt before session messages. For imported sessions, that prompt is not
part of the external conversation and should not be added. Current extraction
also allows `SESSION` scope, which is not appropriate for imported evidence
containers.

Target behavior:

- Ordinary sessions keep existing extraction prefix behavior.
- Import sessions do not prepend Alpha runtime system prompt.
- Import sessions do not introduce local summary or compressed context.
- Import sessions replay imported source messages only, in persisted file order,
  then append an import-aware extraction instruction.
- Imported `role=system` messages remain part of the imported message sequence
  and are replayed as real LLM `system` messages. They are historical external
  transcript messages, not Alpha runtime prompts.
- If an LLM provider cannot replay a supported imported message shape, extraction
  should fail for that source window rather than rewriting the stored import
  transcript.
- Import-aware extraction does not allow `SESSION` scope and does not include
  the import session as an allowed `about` reference.
- Import-aware extraction must not treat assistant output as user
  self-description unless a user message adopts, corrects, or otherwise makes
  the assistant output evidence.
- The general background extraction contract remains unchanged for ordinary
  sessions.

### Cognition Source Time

Imported messages preserve exact source instants in persistent message records,
but cognition uses a simpler approximate source time.

Rules:

- Extraction source windows expose a single approximate `source_time`, not a
  time range and not per-draft supporting message ids.
- Approximate `source_time` is the latest `created_at` UTC instant among the
  source messages selected for that extraction window.
- Imported `system`, `user`, `assistant`, and `tool` source messages all
  participate in approximate `source_time`.
- Alpha-generated `system_reminder` and `compressed_message` records do not
  participate in approximate `source_time`.
- Prompt rendering may show `source_time` in the hidden session timezone and may
  round display to minute precision.
- Consolidation and summary prompts should prefer `source_time` over
  `held_since` when reasoning about recency.
- `source_time` is not an audit field. Exact message timestamps remain available
  through the stored source messages and import mapping records.

### Processing Priority

Imported sessions should be distinguishable from ordinary inactive sessions.

Desired scheduling behavior:

- Foreground/current sessions remain highest priority.
- Ordinary recent inactive sessions come before bulk imported history.
- Imported sessions are lower priority and can be rate-limited later.
- Among imported sessions, extraction selects conversations in ascending
  earliest pending source message time order (oldest first), so consolidation
  supersede chains terminate at the user's most recent expression instead of an
  older conversation arriving late and "contradicting" newer state.

First version can use the existing worker flow with import-session detection,
but must not let imported history break ordinary chat responsiveness.
It does not need an import-specific bypass around ordinary inactive-session
eligibility; historical imports are expected to satisfy existing inactivity
rules in normal use.

For imported sessions, "oldest first" means sorting by the earliest pending
extractable source message `created_at` UTC instant, with a stable id tie-breaker.
Conversation-level timestamps are not used for this ordering.

### Status Semantics

`import status` tracks import and extraction progress.

First version status scope:

- Import batch existence and aggregate counts.
- Number of conversations/messages seen.
- Number of conversations/messages newly created.
- Number of messages deduped/skipped.
- Extraction progress over imported `session_message` sources. Default status
  counts messages; verbose status may aggregate those message counts by
  conversation.
- Imported `system`, `user`, `assistant`, and `tool` source messages all count
  toward extraction progress. Deduped messages do not, because they create no
  new `session_messages`.

Do not include consolidation/profile-summary completion in first-version status.
That would make status semantics harder to explain and verify.

## CLI Surface

Target commands:

```bash
alpha cognition import conversations path/to/conversations.json
alpha cognition import conversations path/to/conversations.json --dry-run
alpha cognition import status <batch_id>
alpha cognition import status <batch_id> --verbose
```

Default import output should include:

- batch id, unless dry-run
- source provider
- conversations seen
- messages seen
- conversations created/reused
- messages inserted
- messages deduped
- whether import was queued/available for background cognition

Default status output should include:

- batch id
- import status
- aggregate import counts
- extraction pending/claimed/processed/failed/skipped counts

Verbose status can include:

- external conversation id
- title, if present
- import status per conversation
- message inserted/deduped counts per conversation when those attempt-level
  details were persisted for the batch
- internal hidden session id for troubleshooting
- extraction counts per conversation

Default status should not show hidden session ids.

## Validation Strategy

File-size validation happens in the CLI before IPC. Structural validation
happens in the daemon service.

Rules:

- Reject files larger than the first-version size limit before sending them to
  the daemon.
- Parse the entire JSON payload before writing.
- Validate the entire normalized structure before writing.
- Default behavior is whole-batch failure for invalid input.
- Deduped messages are not validation failures.
- First version does not offer `--skip-invalid`.
- First version does not compare duplicate external ids for content mismatch.
- Error paths must point to exact payload locations where possible.

Validation categories:

- Payload too large.
- Malformed JSON.
- Invalid top-level object.
- Missing provider.
- Invalid top-level timezone.
- Empty conversations.
- Missing or duplicate conversation ids.
- Missing or duplicate message ids inside a conversation.
- Invalid role.
- Invalid timestamp, including any explicit timestamp without timezone/offset.
- Message timestamps that are not strictly increasing by UTC instant in a new
  conversation.
- New messages for an existing conversation that are not strictly later than
  the latest already imported message after dedup planning.
- Empty content where content is required.
- Invalid tool call shape.
- Tool message without a matching assistant tool call.
- Unsupported reasoning or attachment fields.
- Metadata present but not an object.

All validation and write-plan errors are whole-batch failures. First version
does not partially import valid conversations from an invalid batch.

## Security And Data Boundary Notes

This is a local personal-agent feature, but boundary discipline still matters.

Rules:

- Do not persist local machine-specific absolute input paths.
- `input_name` may be a filename or user-facing label.
- Imported conversation content is owner-supplied trusted source material in
  first version.
- Do not treat metadata as trusted cognition semantics.
- Do not run tools from imported tool records.
- Imported tool calls/results are historical text/context only.
- Do not let imported sessions become live chat sessions.
- Do not let import IPC fall back to local direct writes when daemon is down.

## Implementation Phases

### Phase 1: Data Contract And Store Foundation

**Goal:** Add the data structures needed for import without exposing CLI yet.

Tasks:

- [x] Define import domain models for batch, conversation, message, validation
      result, import summary, and status summary.
- [x] Extend persistent session message role support for `system`.
- [x] Add import tables to the target schema.
- [x] Add store methods to create/reuse imported conversations and map imported
      messages to session messages.
- [x] Store hidden import session timezone and external-history session
      created/updated times.
- [x] Add store methods to detect import sessions by session id.
- [x] Add store status queries for import batches and imported session messages.
- [x] Ensure imported message content is stored once in `session_messages`.

Acceptance criteria:

- [x] A new database initialized from the target schema accepts persistent
      `system` session messages.
- [x] Imported conversation mappings can identify hidden import sessions.
- [x] Duplicate external message ids can be skipped without duplicate
      `session_messages`.
- [x] Hidden import sessions can preserve external-history created/updated times
      and derived fixed-offset timezones.
- [x] Store tests cover create, reuse, dedup, and import-session detection.

Verification:

- [x] Run focused store/import tests.
- [x] Run existing session context tests affected by role changes.

### Phase 2: Import Service Validation And Planning

**Goal:** Build daemon-side import parsing, validation, dry-run, and write plan
logic without IPC wiring yet.

Tasks:

- [x] Implement normalized JSON parser in service/application layer.
- [x] Treat payloads that reached the service as already CLI size-checked; add a
      defensive service-side size rejection only if it fits the existing IPC
      boundary cleanly.
- [x] Validate required fields and rejected fields.
- [x] Validate optional top-level timezone.
- [x] Validate message timestamps are timezone-aware.
- [x] Validate per-conversation message timestamps are strictly increasing by
      UTC instant.
- [x] Validate role-specific rules for `system`, `user`, `assistant`, `tool`.
- [x] Validate assistant tool calls and tool result matching.
- [x] Produce path-aware validation errors.
- [x] Produce dry-run plan counts without writes.
- [x] Produce real import write plans with conversation reuse and message dedup.
- [x] For existing conversations, dedup first and reject planned inserts that
      would require historical middle insertion or non-increasing append time.

Acceptance criteria:

- [x] Invalid payloads return stable error codes and path details.
- [x] Dry-run returns counts and does not write batches or messages.
- [x] A valid payload produces a deterministic plan for inserted and deduped
      messages.
- [x] Invalid timestamp ordering fails the whole batch in dry-run and real
      planning.

Verification:

- [x] Unit tests for parser and validator.
- [x] Unit tests for dry-run behavior.
- [x] Unit tests for dedup planning.

### Phase 3: Import Write Path

**Goal:** Persist a valid import through the service.

Tasks:

- [x] Generate a new import batch id for every real import attempt.
- [x] Create or reuse imported conversation mappings.
- [x] Generate normal internal session ids for first-time imported
      conversations.
- [x] Bind imported sessions to the main user counterpart.
- [x] Append new messages to hidden sessions in file order.
- [x] Set `session_messages.created_at` from external message `created_at`.
- [x] Keep `session_messages.updated_at` null for new imports.
- [x] Set hidden import session `created_at`, `updated_at`, and `timezone`
      from the import time model.
- [x] Persist message mapping records for inserted messages.
- [x] Count deduped messages.
- [x] Persist aggregate batch status.
- [x] Avoid writing cognitive events, runtime traces, compressed messages, or
      session time reminders.

Acceptance criteria:

- [x] Real import writes batch, conversation, imported message, session message,
      and counterpart binding records.
- [x] Re-importing the same payload creates a new batch with dedup counts and no
      duplicate session messages.
- [x] Re-importing an existing conversation with new messages appends only new
      messages.
- [x] Re-importing an existing conversation rejects new messages that are not
      strictly later than the already imported conversation history.
- [x] Imported sessions are detectable as hidden import sessions.

Verification:

- [x] Integration tests over a temporary SQLite database.
- [x] Existing cognition/store tests still pass for normal sessions.

### Phase 4: Daemon IPC Boundary

**Goal:** Make import daemon-owned and available to clients through IPC.

Tasks:

- [x] Add `conversation_import` request type.
- [x] Add `conversation_import_status` request type.
- [x] Extend daemon request validation for import payload fields.
- [x] Route import requests to the daemon-side service.
- [x] Add service-level import serialization or equivalent concurrency guard.
- [x] Ensure import requests do not use normal turn guard.
- [x] Return structured summaries and validation errors.
- [x] Keep daemon status/stop behavior unchanged.

Acceptance criteria:

- [x] IPC import dry-run returns summary without writes.
- [x] IPC import real run writes through the daemon-owned service.
- [x] IPC status returns aggregate status.
- [x] Unknown or malformed import requests return protocol errors.
- [x] Ordinary `ask` and `chat_turn` behavior remains unchanged.

Verification:

- [x] Daemon model/protocol tests.
- [x] Daemon runtime IPC tests.

### Phase 5: CLI Commands

**Goal:** Add user-facing commands that call daemon IPC.

Tasks:

- [x] Add `alpha cognition import conversations <file>`.
- [x] Add `--dry-run`.
- [x] Add client-side file size check.
- [x] Send file content and `input_name` through IPC.
- [x] Render import summary.
- [x] Add `alpha cognition import status <batch_id>`.
- [x] Add `--verbose` for conversation-level status.
- [x] Ensure daemon unavailable errors tell the user to start daemon.
- [x] Do not add CLI direct DB write fallback.
- [x] Do not add CLI JSON output in first version.

Acceptance criteria:

- [x] CLI rejects files larger than 50 MB before sending an IPC request.
- [x] CLI dry-run displays validation/plan summary.
- [x] CLI real import displays batch id and counts.
- [x] CLI status displays aggregate extraction progress.
- [x] CLI verbose status displays conversation-level details.
- [x] CLI does not persist absolute input paths.

Verification:

- [x] CLI tests with daemon IPC mocked or test daemon.
- [ ] Manual smoke test with a small normalized JSON file.

### Phase 6: Chat Isolation

**Goal:** Ensure hidden import sessions cannot leak into normal chat.

Tasks:

- [x] Update ordinary session listing/loading paths to exclude import sessions.
- [x] Reject `ask` or `chat` when the requested session id is an import session.
- [x] Add runtime-level guard in ordinary respond path.
- [x] Ensure gateway/API ordinary turn paths also reject import sessions if they
      reach the runtime.
- [x] Keep import/status paths able to reference import sessions where needed,
      and reject debug prompt rendering for import sessions.

Acceptance criteria:

- [x] Import sessions do not appear in ordinary chat history or session choices.
- [x] Direct `chat --session <import_session_id>` is rejected.
- [x] Direct `debug prompt --session <import_session_id>` is rejected before
      prompt rendering.
- [x] Direct daemon turn request with an import session id is rejected.
- [x] Cognition workers can still process import session messages.

Verification:

- [x] CLI tests for hidden sessions.
- [x] Runtime tests for rejection.
- [x] Gateway/daemon tests where relevant.

### Phase 7: Cognition Extraction Integration

**Goal:** Let extraction process imported sessions without adding the local
Alpha runtime system prompt and without producing session-scoped memories.

Tasks:

- [x] Make extraction candidate building detect import sessions through the
      import mapping table.
- [x] For import sessions, build prompt prefix from imported source messages
      only.
- [x] Preserve imported `system` messages in the source sequence.
- [x] Replay imported `system` messages as historical LLM `system` messages.
- [x] Do not prepend Alpha runtime system prompt for import sessions.
- [x] Do not include local summary snapshots or compressed handover context for
      import sessions.
- [x] Use an import-aware extraction instruction that excludes `SESSION` scope.
- [x] Build import-session allowed `about` refs without a session reference.
- [x] Instruct extraction that assistant output is evidence about the user only
      when adopted, corrected, or otherwise made evidence by user messages.
- [x] Keep ordinary session extraction prefix and instruction unchanged.
- [x] Select imported-session extraction candidates in ascending earliest
      pending source message time order (oldest first).
- [x] Attach approximate `source_time` from the latest selected source message
      time to extraction metadata and prompts.

Acceptance criteria:

- [x] Ordinary session extraction still includes existing runtime prefix.
- [x] Import session extraction does not include Alpha runtime system prompt.
- [x] Import session extraction cannot emit `SESSION` scope memory.
- [x] Import session extraction does not convert standalone assistant output
      into user self-description.
- [x] Imported sessions are extracted oldest-first by earliest pending source
      message time.
- [x] Imported `system/user/assistant/tool` messages are converted into LLM
      source messages as intended.
- [x] Import extraction source time is approximate, single-valued, and derived
      from selected source message timestamps.
- [x] Import extraction source refs still use existing background ledger source
      tracking.

Verification:

- [x] Memory extraction worker tests for ordinary and import session prompt,
      prefix, and allowed-scope behavior.
- [x] Status tests that aggregate extraction progress for imported messages.

### Phase 8: Documentation And Final Validation

**Goal:** Document user-facing contract and verify the end-to-end feature.

Tasks:

- [x] Document normalized JSON contract.
- [x] Document CLI commands and daemon requirement.
- [x] Document first-version limitations.
- [x] Document that existing local databases may need rebuild because no
      compatibility migration is provided.
- [x] Run lint, type check, and tests.
- [x] Run end-to-end smoke import through daemon IPC.

Acceptance criteria:

- [x] Documentation explains how to prepare an import file.
- [x] Documentation explains hidden session behavior.
- [x] Documentation explains status meaning.
- [x] Validation commands pass.

Verification:

- [x] `uv run ruff check .`
- [x] `uv run mypy src tests`
- [x] `uv run pytest -q`

## Rollout Guidance

Run the rollout in two stages after implementation lands.

### Pilot Batch Before Bulk Import

- Import a small pilot batch (10-20 conversations) first.
- Manually review the extracted drafts and consolidation decisions produced
  from the pilot before importing full history.
- Verify on pilot output that external assistant statements were not extracted
  as user self-description unless the user adopted or responded to them.
- Tune source data or extraction behavior before bulk import if pilot quality
  is poor; batch idempotency makes incremental rollout safe.

### Backlog Watch During Bulk Import

- Watch `background_source_progress` pending depth for the `extraction` and
  `consolidation` stages during and after bulk import.
- Sustained consolidation backlog or visibly duplicated active beliefs after
  bulk import is the trigger signal to execute
  `docs/todo/consolidation_reconciliation_plan.md`. Do not start that plan on
  a calendar basis; start it when this signal appears.

## Dependency Order

Recommended sequence:

1. Schema/model/store foundation.
2. Service parser/validator/dry-run.
3. Service write path.
4. Daemon IPC.
5. CLI.
6. Chat isolation guards.
7. Extraction prompt special case.
8. Docs and final verification.

Rationale:

- Store and domain models are prerequisites for service and status.
- Service must own validation before daemon/CLI expose it.
- IPC comes before CLI because CLI must not write directly.
- Chat isolation guards rely on import-session detection and are implemented as
  their own verification phase.
- Extraction prompt behavior depends on import-session detection.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Imported sessions leak into normal chat | High | Use import mapping table as authoritative session classifier and reject at runtime. |
| Import extraction creates session-scoped memories | High | Use import-aware extraction instructions and validation context that exclude `SESSION` scope and session `about` refs. |
| Assistant statements become user self-description without user adoption | High | Import-aware extraction must treat assistant output as evidence only when user messages adopt, correct, or otherwise make it evidence. |
| Large imports block normal daemon work | Medium | Enforce 50 MB CLI file limit, parse before write transaction, serialize import writes only. |
| Duplicate imports create duplicate memories | High | External message id is required; dedup before writing session messages. |
| Existing local DB has old CHECK constraints | Medium | No compatibility migration; document local state rebuild requirement. |
| Tool messages cannot replay because of missing tool calls | Medium | Require tool_call_id and assistant tool_calls with matching ids. |
| Naive or non-increasing timestamps corrupt transcript order | High | Require timezone-aware message timestamps and strict UTC-instant increase within each conversation. |
| Historical middle insertion corrupts imported session replay | High | Dedup first on re-import, then reject new messages that are not strictly later than existing imported history. |
| Out-of-order extraction lets older expressions supersede newer ones | Medium | Extract imported sessions oldest-first by earliest pending source message time so supersede chains end at the most recent expression. |
| First version status overpromises cognition completion | Medium | Track only import completion and extraction source progress. |

## Settled First-Version Time Decisions

- Source message time is available to cognition as a single approximate
  `source_time`.
- Approximate `source_time` is derived from selected source message
  `created_at`, not import processing time.
- `held_since` and `observed_at` remain Alpha processing/holding time.
- Exact message timestamps remain in persistent source records and import
  mapping records.
- Per-draft supporting message ids and source-time ranges are not required in
  the first version.

Low-level details not explicitly specified in this plan should follow existing
project patterns and the architectural decisions above.

## First-Version Acceptance Checklist

- [x] Valid normalized JSON import succeeds through daemon IPC.
- [x] Dry-run validates and reports planned counts without writes.
- [x] Re-import creates a new batch and dedups existing messages.
- [x] Imported messages persist with original timestamps.
- [x] Imported message timestamps are timezone-aware and strictly increasing
      within each conversation by UTC instant.
- [x] Existing conversation re-import rejects planned inserts that are not
      strictly later than existing imported history.
- [x] Hidden import session timezone is top-level `timezone` when provided, or
      the fixed offset from the first message timestamp otherwise.
- [x] Imported `system` messages are persistable.
- [x] Existing Alpha runtime system prompt remains non-persistent.
- [x] Import does not generate session time reminders.
- [x] Imported conversations map to hidden internal sessions.
- [x] Hidden import sessions are invisible and not continuable in normal chat.
- [x] Hidden import sessions cannot be rendered through `debug prompt`.
- [x] Imported sessions bind to main user counterpart.
- [x] Import does not write cognitive events.
- [x] Import does not write runtime traces.
- [x] Import does not generate compressed messages.
- [x] Import status reports aggregate import counts.
- [x] Import status reports extraction progress for imported session messages.
- [x] Import extraction does not prepend Alpha runtime system prompt.
- [x] Import extraction replays imported `system` messages as historical LLM
      `system` messages.
- [x] Import extraction exposes approximate source time derived from selected
      source messages.
- [x] Import extraction does not produce `SESSION` scope memories.
- [x] Import extraction does not treat standalone assistant output as user
      self-description.
- [x] First-version limitations are documented.
