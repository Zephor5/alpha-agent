# External Conversation Import Execution Plan

## Status

Planned.

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

- Use external message `created_at` as the persisted session message time.
- Do not replace original message time with import time.
- Store import time separately in import batch/message records.
- Target cognition behavior should have access to source message time, not just
  processing time, when extracting and consolidating memories. The exact
  project-wide representation of message-time evidence remains an open design
  detail because current background cognition also loses accurate per-message
  time information.
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
- `created_at`: required valid timestamp.
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

### Optional Fields

Top level:

- `metadata`: optional JSON object.

Conversation:

- `title`: optional string.
- `created_at`: optional timestamp.
- `updated_at`: optional timestamp.
- `metadata`: optional JSON object.

Message:

- `metadata`: optional JSON object.

### Rejected Fields And Cases

First version rejects:

- Missing or empty message content, except assistant tool-call messages.
- Roles outside `system`, `user`, `assistant`, `tool`.
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
- Do not compare existing content for conflicts in first version.
- Do not overwrite existing messages.
- If the same conversation contains new message ids, append them to the same
  hidden import session.
- Message append order follows file order.
- Do not sort by timestamp.
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
- Plan writes and dedup counts.
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
      "message": "created_at must be a valid timestamp"
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
- Import sessions replay imported source messages only, then an import-aware
  extraction instruction.
- Imported `role=system` messages remain part of the imported message sequence.
- Import-aware extraction does not allow `SESSION` scope and does not include
  the import session as an allowed `about` reference.
- Import-aware extraction must not treat assistant output as user
  self-description unless a user message adopts, corrects, or otherwise makes
  the assistant output evidence.
- The general background extraction contract remains unchanged for ordinary
  sessions.

### Processing Priority

Imported sessions should be distinguishable from ordinary inactive sessions.

Desired scheduling behavior:

- Foreground/current sessions remain highest priority.
- Ordinary recent inactive sessions come before bulk imported history.
- Imported sessions are lower priority and can be rate-limited later.
- Among imported sessions, extraction selects conversations in ascending
  original-time order (oldest first), so consolidation supersede chains
  terminate at the user's most recent expression instead of an older
  conversation arriving late and "contradicting" newer state.

First version can use the existing worker flow with import-session detection,
but must not let imported history break ordinary chat responsiveness.

### Status Semantics

`import status` tracks import and extraction progress.

First version status scope:

- Import batch existence and aggregate counts.
- Number of conversations/messages seen.
- Number of conversations/messages newly created.
- Number of messages deduped/skipped.
- Extraction progress over imported `session_message` sources.

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
- Empty conversations.
- Missing or duplicate conversation ids.
- Missing or duplicate message ids inside a conversation.
- Invalid role.
- Invalid timestamp.
- Empty content where content is required.
- Invalid tool call shape.
- Tool message without a matching assistant tool call.
- Unsupported reasoning or attachment fields.
- Metadata present but not an object.

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

- [ ] Define import domain models for batch, conversation, message, validation
      result, import summary, and status summary.
- [ ] Extend persistent session message role support for `system`.
- [ ] Add import tables to the target schema.
- [ ] Add store methods to create/reuse imported conversations and map imported
      messages to session messages.
- [ ] Add store methods to detect import sessions by session id.
- [ ] Add store status queries for import batches and imported session messages.
- [ ] Ensure imported message content is stored once in `session_messages`.

Acceptance criteria:

- [ ] A new database initialized from the target schema accepts persistent
      `system` session messages.
- [ ] Imported conversation mappings can identify hidden import sessions.
- [ ] Duplicate external message ids can be skipped without duplicate
      `session_messages`.
- [ ] Store tests cover create, reuse, dedup, and import-session detection.

Verification:

- [ ] Run focused store/import tests.
- [ ] Run existing session context tests affected by role changes.

### Phase 2: Import Service Validation And Planning

**Goal:** Build daemon-side import parsing, validation, dry-run, and write plan
logic without IPC wiring yet.

Tasks:

- [ ] Implement normalized JSON parser in service/application layer.
- [ ] Treat payloads that reached the service as already CLI size-checked; add a
      defensive service-side size rejection only if it fits the existing IPC
      boundary cleanly.
- [ ] Validate required fields and rejected fields.
- [ ] Validate role-specific rules for `system`, `user`, `assistant`, `tool`.
- [ ] Validate assistant tool calls and tool result matching.
- [ ] Produce path-aware validation errors.
- [ ] Produce dry-run plan counts without writes.
- [ ] Produce real import write plans with conversation reuse and message dedup.

Acceptance criteria:

- [ ] Invalid payloads return stable error codes and path details.
- [ ] Dry-run returns counts and does not write batches or messages.
- [ ] A valid payload produces a deterministic plan for inserted and deduped
      messages.

Verification:

- [ ] Unit tests for parser and validator.
- [ ] Unit tests for dry-run behavior.
- [ ] Unit tests for dedup planning.

### Phase 3: Import Write Path

**Goal:** Persist a valid import through the service.

Tasks:

- [ ] Generate a new import batch id for every real import attempt.
- [ ] Create or reuse imported conversation mappings.
- [ ] Generate normal internal session ids for first-time imported
      conversations.
- [ ] Bind imported sessions to the main user counterpart.
- [ ] Append new messages to hidden sessions in file order.
- [ ] Set `session_messages.created_at` from external message `created_at`.
- [ ] Keep `session_messages.updated_at` null for new imports.
- [ ] Persist message mapping records for inserted messages.
- [ ] Count deduped messages.
- [ ] Persist aggregate batch status.
- [ ] Avoid writing cognitive events, runtime traces, or compressed messages.

Acceptance criteria:

- [ ] Real import writes batch, conversation, imported message, session message,
      and counterpart binding records.
- [ ] Re-importing the same payload creates a new batch with dedup counts and no
      duplicate session messages.
- [ ] Re-importing an existing conversation with new messages appends only new
      messages.
- [ ] Imported sessions are detectable as hidden import sessions.

Verification:

- [ ] Integration tests over a temporary SQLite database.
- [ ] Existing cognition/store tests still pass for normal sessions.

### Phase 4: Daemon IPC Boundary

**Goal:** Make import daemon-owned and available to clients through IPC.

Tasks:

- [ ] Add `conversation_import` request type.
- [ ] Add `conversation_import_status` request type.
- [ ] Extend daemon request validation for import payload fields.
- [ ] Route import requests to the daemon-side service.
- [ ] Add service-level import serialization or equivalent concurrency guard.
- [ ] Ensure import requests do not use normal turn guard.
- [ ] Return structured summaries and validation errors.
- [ ] Keep daemon status/stop behavior unchanged.

Acceptance criteria:

- [ ] IPC import dry-run returns summary without writes.
- [ ] IPC import real run writes through the daemon-owned service.
- [ ] IPC status returns aggregate status.
- [ ] Unknown or malformed import requests return protocol errors.
- [ ] Ordinary `ask` and `chat_turn` behavior remains unchanged.

Verification:

- [ ] Daemon model/protocol tests.
- [ ] Daemon runtime IPC tests.

### Phase 5: CLI Commands

**Goal:** Add user-facing commands that call daemon IPC.

Tasks:

- [ ] Add `alpha cognition import conversations <file>`.
- [ ] Add `--dry-run`.
- [ ] Add client-side file size check.
- [ ] Send file content and `input_name` through IPC.
- [ ] Render import summary.
- [ ] Add `alpha cognition import status <batch_id>`.
- [ ] Add `--verbose` for conversation-level status.
- [ ] Ensure daemon unavailable errors tell the user to start daemon.
- [ ] Do not add CLI direct DB write fallback.
- [ ] Do not add CLI JSON output in first version.

Acceptance criteria:

- [ ] CLI rejects files larger than 50 MB before sending an IPC request.
- [ ] CLI dry-run displays validation/plan summary.
- [ ] CLI real import displays batch id and counts.
- [ ] CLI status displays aggregate extraction progress.
- [ ] CLI verbose status displays conversation-level details.
- [ ] CLI does not persist absolute input paths.

Verification:

- [ ] CLI tests with daemon IPC mocked or test daemon.
- [ ] Manual smoke test with a small normalized JSON file.

### Phase 6: Chat Isolation

**Goal:** Ensure hidden import sessions cannot leak into normal chat.

Tasks:

- [ ] Update ordinary session listing/loading paths to exclude import sessions.
- [ ] Reject `ask` or `chat` when the requested session id is an import session.
- [ ] Add runtime-level guard in ordinary respond path.
- [ ] Ensure gateway/API ordinary turn paths also reject import sessions if they
      reach the runtime.
- [ ] Keep debug/import/status paths able to reference import sessions where
      needed.

Acceptance criteria:

- [ ] Import sessions do not appear in ordinary chat history or session choices.
- [ ] Direct `chat --session <import_session_id>` is rejected.
- [ ] Direct daemon turn request with an import session id is rejected.
- [ ] Cognition workers can still process import session messages.

Verification:

- [ ] CLI tests for hidden sessions.
- [ ] Runtime tests for rejection.
- [ ] Gateway/daemon tests where relevant.

### Phase 7: Cognition Extraction Integration

**Goal:** Let extraction process imported sessions without adding the local
Alpha runtime system prompt and without producing session-scoped memories.

Tasks:

- [ ] Make extraction candidate building detect import sessions through the
      import mapping table.
- [ ] For import sessions, build prompt prefix from imported source messages
      only.
- [ ] Preserve imported `system` messages in the source sequence.
- [ ] Do not prepend Alpha runtime system prompt for import sessions.
- [ ] Do not include local summary snapshots or compressed handover context for
      import sessions.
- [ ] Use an import-aware extraction instruction that excludes `SESSION` scope.
- [ ] Build import-session allowed `about` refs without a session reference.
- [ ] Instruct extraction that assistant output is evidence about the user only
      when adopted, corrected, or otherwise made evidence by user messages.
- [ ] Keep ordinary session extraction prefix and instruction unchanged.
- [ ] Select imported-session extraction candidates in ascending original
      conversation time order (oldest first).

Acceptance criteria:

- [ ] Ordinary session extraction still includes existing runtime prefix.
- [ ] Import session extraction does not include Alpha runtime system prompt.
- [ ] Import session extraction cannot emit `SESSION` scope memory.
- [ ] Import session extraction does not convert standalone assistant output
      into user self-description.
- [ ] Imported sessions are extracted oldest-first by original conversation
      time.
- [ ] Imported `system/user/assistant/tool` messages are converted into LLM
      source messages as intended.
- [ ] Import extraction source refs still use existing background ledger source
      tracking.

Verification:

- [ ] Memory extraction worker tests for ordinary and import session prompt,
      prefix, and allowed-scope behavior.
- [ ] Status tests that aggregate extraction progress for imported messages.

### Phase 8: Documentation And Final Validation

**Goal:** Document user-facing contract and verify the end-to-end feature.

Tasks:

- [ ] Document normalized JSON contract.
- [ ] Document CLI commands and daemon requirement.
- [ ] Document first-version limitations.
- [ ] Document that existing local databases may need rebuild because no
      compatibility migration is provided.
- [ ] Run lint, type check, and tests.
- [ ] Run end-to-end smoke import through daemon IPC.

Acceptance criteria:

- [ ] Documentation explains how to prepare an import file.
- [ ] Documentation explains hidden session behavior.
- [ ] Documentation explains status meaning.
- [ ] Validation commands pass.

Verification:

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src tests`
- [ ] `uv run pytest -q`

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
| Out-of-order extraction lets older expressions supersede newer ones | Medium | Extract imported sessions oldest-first by original time so supersede chains end at the most recent expression. |
| First version status overpromises cognition completion | Medium | Track only import completion and extraction source progress. |

## Open Decisions

### Source-Time Propagation Through Cognition

Imported messages persist original `created_at`, but current background
cognition primarily uses processing time when materializing extracted beliefs.
Before relying on time-based supersession quality, decide how source message
time should be represented in extraction prompts, validation context, belief
validity, and consolidation inputs. This likely affects ordinary sessions too,
not only external imports.

Low-level details not explicitly specified in this plan should follow existing
project patterns and the architectural decisions above.

## First-Version Acceptance Checklist

- [ ] Valid normalized JSON import succeeds through daemon IPC.
- [ ] Dry-run validates and reports planned counts without writes.
- [ ] Re-import creates a new batch and dedups existing messages.
- [ ] Imported messages persist with original timestamps.
- [ ] Imported `system` messages are persistable.
- [ ] Existing Alpha runtime system prompt remains non-persistent.
- [ ] Imported conversations map to hidden internal sessions.
- [ ] Hidden import sessions are invisible and not continuable in normal chat.
- [ ] Imported sessions bind to main user counterpart.
- [ ] Import does not write cognitive events.
- [ ] Import does not write runtime traces.
- [ ] Import does not generate compressed messages.
- [ ] Import status reports aggregate import counts.
- [ ] Import status reports extraction progress for imported session messages.
- [ ] Import extraction does not prepend Alpha runtime system prompt.
- [ ] Import extraction does not produce `SESSION` scope memories.
- [ ] Import extraction does not treat standalone assistant output as user
      self-description.
- [ ] First-version limitations are documented.
