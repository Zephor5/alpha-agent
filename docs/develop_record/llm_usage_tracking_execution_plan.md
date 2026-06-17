# LLM Usage Tracking Execution Plan

Date: 2026-06-17
Status: Implemented

## 1. Goal

Add token usage accounting at two different levels:

- `llm_calls`: a global successful-LLM-call usage ledger. It records every successful LLM call the system makes, whether or not the call belongs to a user session.
- `sessions`: session-level usage summary for calls directly involved in continuing that session.

These two surfaces are intentionally not the same model. `llm_calls.session_id` is optional context only; session cumulative fields are updated by the session runtime path and must not be inferred from `llm_calls WHERE session_id = ?`.

This plan is based on active code only. Historical archive material under `docs/develop_record/` was intentionally not used.

## 2. Current Code Facts

`src/alpha_agent/llm/base.py`

- `LLMResponse` currently has no first-class usage field.
- `openai_compatible_response()` receives the full provider payload and already keeps provider payloads in metadata through the shared chat-completions path.

`src/alpha_agent/llm/chat_completions.py`

- MiMo, DeepSeek, and generic OpenAI-compatible providers all pass through `complete_chat_completions()`.
- This makes the shared OpenAI-compatible response normalizer the right place to normalize MiMo and DeepSeek usage.

`src/alpha_agent/llm/codex.py`

- Codex builds `LLMResponse` directly from a Responses-style payload.
- If Codex returns a compatible `usage` object, it needs separate normalization in this provider path.

`src/alpha_agent/runtime/agent.py`

- Foreground answer-path calls go through `_call_model()`.
- `_call_model()` already creates an `llm_call_id`, writes `llm.started` / `llm.completed` traces, and returns trace ids through `RetriedLLMCompletion`.
- Tool-call assistant messages and final assistant messages are persisted as `session_messages`.

`src/alpha_agent/runtime/context_handover.py`

- `compress_session_context()` calls the provider directly and writes `handover_compression.*` traces.
- It currently does not use the same `llm_call_id` / accounting path as foreground answer calls.

`src/alpha_agent/cognition/loops/workers/*` and `src/alpha_agent/cognition/loops/feedback_attribution.py`

- Background cognition and feedback attribution calls mostly use `traced_llm_complete()`.
- These calls should be written to `llm_calls`, but must not update `sessions` token counters.

`src/alpha_agent/state/schema.sql`, `src/alpha_agent/state/models.py`, `src/alpha_agent/state/store.py`

- `sessions` currently stores only identity/timezone/timestamps.
- There is no durable LLM usage ledger.

`src/alpha_agent/runtime/context_budget.py`

- The existing estimator returns `message_tokens`, `tool_schema_tokens`, output reserve, safety margin, and total budget usage.
- For `occupied_tokens` estimates, only `message_tokens + tool_schema_tokens` should be used.

## 3. Usage Contract

### 3.1 `LLMResponse.usage`

Add a provider-neutral usage object:

```python
@dataclass(frozen=True)
class LLMUsage:
    total_tokens: int
    cached_tokens: int
    prompt_cache_miss_tokens: int
    reasoning_tokens: int
    completion_tokens: int
```

Then add this field to `LLMResponse`:

```python
usage: LLMUsage | None = None
```

Rules:

- `LLMResponse.usage is None` means the provider did not return a usable usage payload.
- If `LLMResponse.usage` exists, all five fields must be non-null integers.
- Do not add `prompt_tokens` to `LLMUsage`; by contract, `prompt_tokens = cached_tokens + prompt_cache_miss_tokens`.

### 3.2 Normalization Rules

Normalize provider usage into:

```json
{
  "total_tokens": 5400,
  "cached_tokens": 4096,
  "prompt_cache_miss_tokens": 1008,
  "reasoning_tokens": 34,
  "completion_tokens": 296
}
```

Supported extraction rules:

| Normalized field | Source |
| --- | --- |
| `total_tokens` | `usage.total_tokens` |
| `completion_tokens` | `usage.completion_tokens` |
| `reasoning_tokens` | `usage.completion_tokens_details.reasoning_tokens`, default `0` when absent |
| `cached_tokens` | `usage.prompt_tokens_details.cached_tokens`, fallback `usage.prompt_cache_hit_tokens`, fallback `0` |
| `prompt_cache_miss_tokens` | `usage.prompt_cache_miss_tokens`, fallback `usage.prompt_tokens - cached_tokens`, fallback `usage.prompt_tokens` when no cache field exists |

Provider shapes outside these supported forms are not a target for this implementation.

### 3.3 Supplied Provider Samples

MiMo raw usage:

```json
{
  "completion_tokens": 296,
  "completion_tokens_details": {"reasoning_tokens": 34},
  "prompt_tokens": 5104,
  "prompt_tokens_details": {"cached_tokens": 4096},
  "total_tokens": 5400
}
```

Normalized:

```json
{
  "total_tokens": 5400,
  "cached_tokens": 4096,
  "prompt_cache_miss_tokens": 1008,
  "reasoning_tokens": 34,
  "completion_tokens": 296
}
```

DeepSeek raw usage:

```json
{
  "completion_tokens": 104,
  "completion_tokens_details": {"reasoning_tokens": 96},
  "prompt_cache_hit_tokens": 256,
  "prompt_cache_miss_tokens": 106,
  "prompt_tokens": 362,
  "prompt_tokens_details": {"cached_tokens": 256},
  "total_tokens": 466
}
```

Normalized:

```json
{
  "total_tokens": 466,
  "cached_tokens": 256,
  "prompt_cache_miss_tokens": 106,
  "reasoning_tokens": 96,
  "completion_tokens": 104
}
```

## 4. Persistence Design

This project currently directs agents not to preserve compatibility with existing data, so implementation should directly update `schema.sql`, models, and store code. Do not add a gradual migration layer unless that project rule changes.

### 4.1 `sessions` Columns

Add session-level counters:

```sql
total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (total_tokens >= 0),
cached_tokens INTEGER NOT NULL DEFAULT 0 CHECK (cached_tokens >= 0),
prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_cache_miss_tokens >= 0),
reasoning_tokens INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
occupied_tokens INTEGER NOT NULL DEFAULT 0 CHECK (occupied_tokens >= 0)
```

The five cumulative token counters include only successful LLM calls directly involved in continuing the session:

- foreground answer-path LLM calls
- tool-loop LLM calls
- LLM calls that decide to use memory tools
- handover/compression LLM calls

They do not include cognition maintenance calls:

- background memory extraction
- background consolidation
- conflict review
- summary generation
- feedback attribution
- compact extraction worker calls after a handover

If a direct session call has no normalized `LLMResponse.usage`, session cumulative counters do not increase for that call.

### 4.2 `llm_calls` Table

Add a global successful-call ledger:

```sql
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    worker_name TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (total_tokens >= 0),
    cached_tokens INTEGER NOT NULL DEFAULT 0 CHECK (cached_tokens >= 0),
    prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_cache_miss_tokens >= 0),
    reasoning_tokens INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
    completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    raw_usage TEXT NOT NULL DEFAULT '{}',
    started_trace_id TEXT,
    completed_trace_id TEXT,
    created_at TEXT NOT NULL
);
```

Field rules:

- `id` is exactly the runtime `llm_call_id`.
- `session_id` is nullable and contextual only. It does not imply the call contributes to `sessions` counters.
- Do not add a foreign key from `llm_calls.session_id` to `sessions`.
- `worker_name` is nullable and used to group background calls that may not have a session.
- Token columns are non-null. If the provider returns no usage, write `0` for all five token columns.
- `raw_usage` stores only the provider's raw `usage` sub-object. Do not store the full request payload or full response payload in SQLite.
- Keep `started_trace_id` and `completed_trace_id` so a usage row can be traced back to runtime traces.

Recommended indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at
    ON llm_calls(created_at);

CREATE INDEX IF NOT EXISTS idx_llm_calls_session_created
    ON llm_calls(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_llm_calls_worker_created
    ON llm_calls(worker_name, created_at);
```

Do not add these fields to `llm_calls` in the first implementation:

- `call_kind`
- `turn_id`
- `round_name`
- `finish_reason`
- `retry_count`
- `completed_at`
- `prompt_token_estimate`
- `prompt_tokens`
- `occupied_tokens_before`
- `occupied_tokens_after`
- `output_message_id`
- `request_summary`
- `response_summary`
- generic `metadata`

### 4.3 Session Messages

Add `llm_call_id` to assistant and compressed message `provider_metadata`.

Do not duplicate normalized usage into `session_messages.provider_metadata`. Usage belongs in `llm_calls`; session messages only need the call id to support reverse lookup.

### 4.4 Store API

Add state models:

- `LLMUsageRecord`
- `LLMCallRecord`

Add store methods:

- `append_llm_call(...) -> LLMCallRecord`
- `add_session_usage(session_id, usage, conn=...) -> SessionRecord`
- `update_session_occupied_tokens(session_id, occupied_tokens, conn=...) -> SessionRecord`
- `list_llm_calls(...) -> list[LLMCallRecord]`

`llm_calls` and `sessions` counters are separate surfaces. Direct session runtime code should update both when appropriate, but the design does not require strong transactional coupling between them. Usage recording or session-stat write failures should not block a successful response; write a runtime trace and continue.

No compensation or backfill mechanism is required in the first implementation.

## 5. `occupied_tokens` Strategy

### 5.1 Meaning

`sessions.occupied_tokens` means:

The current estimated token occupancy for continuing this session.

It is not a lifetime cost counter. Lifetime cost is represented by the cumulative session token fields.

### 5.2 Normal LLM Calls

For a successful direct session LLM call with normalized usage:

```text
occupied_tokens = LLMResponse.usage.total_tokens
```

This applies to normal foreground answer-path calls and tool-loop LLM calls.

If the provider returns no normalized usage, use the estimate described below.

### 5.3 Handover / Compression

Compression calls are special.

The compression LLM call usage must be added to the session cumulative counters, because it is direct session work. But compression call `total_tokens` must not become `occupied_tokens`, because the compression prompt includes the pre-compression source context that will be removed from the next continuation prompt.

After compression succeeds:

```text
occupied_tokens = estimated next continuation input tokens
```

The estimate should use the actual answer-path prompt construction that would be used after compression:

- default runtime system prompt
- latest compressed message
- ordinary session messages after the compressed message
- current answer-path model tool schemas

It should not include:

- future unknown user message
- output reserve
- safety margin
- separate stable self-memory or counterpart profile reinjection after compression

The last point matches current runtime behavior: stable profile/reminder content may be included inside the compressed message, but it is not reinserted as separate prompt messages after compression.

### 5.4 Tool Truncation

Tool replay truncation changes the prompt shape without producing a new LLM usage value.

After `truncate_tool_context_if_needed()` actually mutates replay payloads:

```text
occupied_tokens = estimated current continuation input tokens
```

Use the same estimation formula below.

Do not add extra `occupied_tokens` refreshes merely because a user message or ordinary tool result message was appended. Those states are immediately followed by normal runtime processing, and the field should be refreshed by LLM usage, compression, or truncation.

### 5.5 Estimation Formula

Use the existing context estimator, but only count model input and tool schema:

```text
estimate = estimate_context_budget(messages, tools=model_tools, ...)
occupied_tokens = estimate.message_tokens + estimate.tool_schema_tokens
```

Do not include:

- `expected_output_reserve_tokens`
- `safety_margin_tokens`

The `messages` input must come from the same runtime prompt path that would be sent to the model. Do not invent theoretical prompt components that the runtime would not actually send.

## 6. Integration Plan

### Phase 1: Normalize `LLMResponse.usage`

Files likely touched:

- `src/alpha_agent/llm/base.py`
- optional new `src/alpha_agent/llm/usage.py`
- `src/alpha_agent/llm/chat_completions.py`
- `src/alpha_agent/llm/codex.py`
- `tests/test_llm_providers.py`

Work:

- Add `LLMUsage`.
- Add `usage: LLMUsage | None = None` to `LLMResponse`.
- Normalize MiMo, DeepSeek, and generic OpenAI-compatible usage in the shared response normalizer.
- Normalize Codex usage when a compatible usage object is present.
- Keep raw provider usage available for `llm_calls.raw_usage`.

Acceptance:

- MiMo sample normalizes to `total=5400`, `cached=4096`, `miss=1008`, `reasoning=34`, `completion=296`.
- DeepSeek sample normalizes to `total=466`, `cached=256`, `miss=106`, `reasoning=96`, `completion=104`.
- Providers without usage return `LLMResponse(..., usage=None)`.
- `LLMUsage` fields are never nullable.

### Phase 2: Add State Models And Schema

Files likely touched:

- `src/alpha_agent/state/schema.sql`
- `src/alpha_agent/state/models.py`
- `src/alpha_agent/state/store.py`
- `tests/test_state_store.py`
- `tests/test_session_context.py`

Work:

- Add session cumulative token columns and `occupied_tokens`.
- Add the minimal `llm_calls` table.
- Extend `SessionRecord`.
- Add `LLMCallRecord`.
- Add store methods for appending call rows, incrementing session usage, updating occupied tokens, and listing calls.

Acceptance:

- A new session starts with all usage counters at `0`.
- `llm_calls.session_id` accepts NULL.
- `llm_calls` token columns are non-null and default to `0`.
- `llm_calls.id` stores the runtime `llm_call_id`.
- Updating `occupied_tokens` can decrease the value and does not mutate cumulative usage counters.

### Phase 3: Foreground Runtime Accounting

Files likely touched:

- `src/alpha_agent/runtime/agent.py`
- optional new `src/alpha_agent/runtime/llm_accounting.py`
- `tests/test_agent_loop.py`

Work:

- Use `_call_model()` as the foreground successful-call accounting point.
- Write one `llm_calls` row for each successful foreground LLM response.
- Use `llm_call_id` as `llm_calls.id`.
- Add `llm_call_id` to assistant message `provider_metadata`.
- For direct session calls with usage, increment session cumulative counters and set `occupied_tokens = usage.total_tokens`.
- For direct session calls without usage, leave cumulative counters unchanged and update `occupied_tokens` with the runtime prompt estimate.
- Keep accounting failures non-blocking and observable through runtime traces.

Acceptance:

- A simple answer turn writes one `llm_calls` row.
- A tool loop writes one `llm_calls` row per successful LLM round.
- Assistant messages contain `provider_metadata.llm_call_id`.
- Session cumulative counters update only for direct session LLM calls with normalized usage.
- Cognition background calls do not update session counters.

### Phase 4: Handover / Compression Accounting

Files likely touched:

- `src/alpha_agent/runtime/context_handover.py`
- `src/alpha_agent/runtime/agent.py`
- `tests/test_context_handover.py`
- `tests/test_agent_loop.py`

Work:

- Give `compress_session_context()` a generated `llm_call_id`.
- Write one `llm_calls` row for successful compression calls.
- Add `llm_call_id` to compressed message `provider_metadata`.
- Add compression usage to session cumulative counters when usage exists.
- After compression, set `occupied_tokens` using the post-compression next-continuation estimate, not compression call `total_tokens`.

Acceptance:

- Pre-user compression writes an LLM call row.
- Tool-loop compression writes an LLM call row.
- Compression usage increments session cumulative counters.
- Compression reduces `occupied_tokens` when the compressed continuation prompt is smaller than the covered pre-compression context.

### Phase 5: Background LLM Call Ledger

Files likely touched:

- `src/alpha_agent/llm/tracing.py`
- `src/alpha_agent/cognition/loops/workers/_common.py`
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/loops/workers/memory_summary.py`
- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- cognition worker tests

Work:

- Ensure background calls receive or generate an `llm_call_id`.
- Write successful background calls to `llm_calls`.
- Set `worker_name` for worker-originated calls.
- Set `session_id` only when it is naturally available; otherwise leave it NULL.
- Do not update `sessions` cumulative counters or `occupied_tokens` from background cognition / feedback / compact extraction calls.

Acceptance:

- Memory extraction, consolidation, summary, conflict review, feedback attribution, and compact extraction successful calls can appear in `llm_calls`.
- Calls without session context write with `session_id = NULL`.
- Background accounting does not affect session usage totals.

### Phase 6: Tool Truncation Occupancy

Files likely touched:

- `src/alpha_agent/runtime/session_context.py`
- `src/alpha_agent/runtime/agent.py`
- `tests/test_session_context.py`
- `tests/test_agent_loop.py`

Work:

- After tool replay payload truncation actually changes stored replay payloads, update `occupied_tokens` by estimating the current runtime continuation prompt.
- Use `estimate.message_tokens + estimate.tool_schema_tokens`.

Acceptance:

- A truncation test shows `occupied_tokens` decreases or otherwise reflects the truncated replay payload.
- No extra refresh is added purely for ordinary user-message or tool-result-message append.

### Phase 7: Documentation Updates

Files likely touched:

- `README.md`
- `AGENTS.md`, only if new modules/tables materially change the project content map

Work:

- Document `LLMResponse.usage`.
- Document the distinction between `llm_calls` global ledger and `sessions` direct-session usage counters.
- Document `occupied_tokens` as current continuation occupancy, not cumulative lifetime spend.
- Document that full raw LLM payloads remain gated by debug JSONL logging, while SQLite stores only raw `usage`.

Acceptance:

- Active docs describe the new tables and semantics.
- No repository file contains machine-specific absolute paths.

## 7. Test Matrix

Run targeted tests during implementation:

```bash
uv run pytest tests/test_llm_providers.py -q
uv run pytest tests/test_state_store.py tests/test_session_context.py tests/test_context_budget.py -q
uv run pytest tests/test_agent_loop.py tests/test_context_handover.py -q
uv run pytest tests/cognition/test_consolidation_loop.py tests/cognition/test_feedback_attribution.py -q
```

Run full validation before completion:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## 8. Main Risks

Duplicate IDs

- Use `llm_calls.id = llm_call_id`.
- Do not create a second independent id for the call row.

Leaking prompt/response content

- Store only raw `usage` in SQLite.
- Keep full request and response payload logging behind existing debug JSONL behavior.

Confusing global call ledger with session totals

- `llm_calls` records all successful calls.
- `sessions` counters record only direct session calls.
- Do not compute session counters by aggregating `llm_calls`.

Incorrect `occupied_tokens` after compression

- Never use the compression call `total_tokens` as post-compression occupancy.
- Estimate the actual next continuation prompt after the compressed message is written.

Layering violation

- `StateStore` should expose persistence methods only.
- Runtime/accounting code should assemble prompts and compute `occupied_tokens`, because it owns the prompt and tool context.
