# Belief Feedback Loop Plan

## Objective

Close the act-feedback-update arc for recalled beliefs: when a runtime turn
uses `memory_recall` results, judge the user's next message against those
beliefs in real time, record verdicts as `RECEIVED_FEEDBACK` cognitive events,
and route them into existing belief governance (feedback history appends and
conflict review).

## Target Behavior

- Recall usage is read from existing session tool messages
  (`provider_metadata["tool_name"] == "memory_recall"`); no new event kind
  records usage.
- When a new user turn starts and the previous assistant turn recalled at
  least one belief, the runtime submits one feedback attribution job.
- The job snapshots the already-built answer prompt messages in the
  foreground and appends a small attribution instruction tail, so the provider
  call shares the main session's prompt prefix. It runs in a daemon thread and
  never blocks, fails, or mutates the main turn.
- Attribution output is contract-validated JSON: exactly one verdict per
  recalled belief id, from `confirmed`, `contradicted`, `corrected`, or
  `irrelevant`. Every non-irrelevant verdict carries an `evidence_quote` that
  must be a verbatim substring of the new user message.
- Each non-irrelevant verdict emits one `RECEIVED_FEEDBACK` cognitive event
  and appends a categorical entry to the belief's `feedback_history`, so a
  challenged belief keeps its challenge record regardless of later review
  outcomes.
- `contradicted` and `corrected` additionally enqueue a `CONFLICT_REVIEW`
  source window consumed by the existing `MemoryConflictReviewWorker`; no new
  resolution logic is added.
- Processing ledger rows on a new `feedback_attribution` stage, keyed by the
  recall tool message, provide idempotency and observability. A row is marked
  succeeded only after every verdict consequence is applied; consequences are
  idempotent, and `RECEIVED_FEEDBACK` emission is at-least-once.
- No numeric confidence, scores, or strength fields anywhere; belief strength
  stays categorical event history.

## Out of Scope

- Attribution for summary-snapshot context (self memory summary, counterpart
  profile).
- Feedback arriving later than the immediate next user message; offline
  catch-up sweeps.
- Tool-outcome attribution for procedure beliefs.
- Metrics reporting CLI.

## Phase 1: Deterministic Helpers and Contracts

### Task 1: Add recall usage parser

Files likely touched:

- `src/alpha_agent/cognition/loops/feedback_attribution.py` (new)
- `src/alpha_agent/cognition/loops/__init__.py`
- `tests/cognition/test_feedback_attribution.py` (new)

Implementation:

- Add `recalled_beliefs_for_previous_turn(store, session_id, before_ordinal)`
  returning ordered, deduplicated recalled belief handles plus their source
  tool message ids.
- Scan only the assistant turn immediately preceding `before_ordinal`: walk
  session messages backwards from `before_ordinal` and stop at the previous
  `user_message` boundary.
- Select `tool_message` rows whose `provider_metadata["tool_name"]` is
  `memory_recall` and parse `results[].id`, `content`, `memory_kind`, and
  `scope` from the replay payload.
- Tolerate payloads rewritten by tool context truncation (truncated string
  values remain valid JSON).
- Return belief handles in first-recalled order with stable dedup by id.

Acceptance criteria:

- [ ] Recall results from the immediately preceding assistant turn are
      returned with ids, contents, and source tool message ids.
- [ ] Recall results from earlier turns are not returned.
- [ ] Non-recall tool messages and empty recall results yield an empty list.
- [ ] Truncated recall tool payloads parse without error.
- [ ] Repeated calls over unchanged state return identical output.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
```

### Task 2: Add attribution output contract

Files likely touched:

- `src/alpha_agent/cognition/background_llm_contract.py`
- `tests/cognition/test_feedback_attribution.py`

Implementation:

- Add `feedback_attribution_output_json_schema` for one JSON object with
  `payload.verdicts` as a non-empty array.
- Each verdict has exactly `belief_id`, `verdict`, and `evidence_quote`;
  `verdict` is one of `confirmed`, `contradicted`, `corrected`, `irrelevant`;
  `evidence_quote` is required and non-empty unless `verdict` is `irrelevant`.
- Add deterministic validation against an attribution context:
  - Every `belief_id` must be in the recalled belief id whitelist.
  - Every whitelisted belief id must appear exactly once.
  - Every non-irrelevant `evidence_quote` must be a verbatim substring of the
    new user message content.
  - Reject ids outside the whitelist, duplicate ids, confidence, scores,
    numeric strength fields, and unknown keys.
- Exempt `evidence_quote` from the contract's prompt-injection string
  rejection: the verbatim-substring check already proves the text came from
  the user rather than being model-invented, and injection-like phrasing in a
  genuine user correction is legitimate evidence. All other output fields keep
  the injection rejection.
- Return typed validated verdict objects.

Acceptance criteria:

- [ ] A valid full-coverage verdict set passes validation.
- [ ] Unknown, missing, or duplicated belief ids are rejected.
- [ ] A non-irrelevant verdict whose quote is not a verbatim substring of the
      user message is rejected.
- [ ] An `evidence_quote` containing injection-like text passes when it is a
      verbatim substring of the user message.
- [ ] Injection-like text in any other output field is still rejected.
- [ ] Confidence, score, and unknown fields are rejected.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
```

### Task 3: Add feedback attribution ledger stage

Files likely touched:

- `src/alpha_agent/cognition/processing_ledger.py`
- `src/alpha_agent/cognition/state_service.py`
- `tests/cognition/test_feedback_attribution.py`

Implementation:

- Add `BackgroundStage.FEEDBACK_ATTRIBUTION`.
- Ledger rows use `source_type="session_message"`, `source_id=<recall tool
  message id>`, `target_unit=f"session:{session_id}"`, and a deterministic
  idempotency key.
- Rows are claimed only after the service has acquired a worker slot, and are
  marked succeeded or failed at job completion; already-claimed or succeeded
  rows block duplicate submission. Saturated submissions claim nothing.
- Extend any stage-specific validation in the state service so the new stage
  is accepted.

Acceptance criteria:

- [ ] Submitting the same recall tool message twice creates one ledger claim.
- [ ] Completion marks all claimed rows succeeded; failures record
      `last_error`.
- [ ] Ledger rows are readable per stage for observability.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
uv run mypy src tests
```

## Phase 2: Realtime Attribution Service

### Task 4: Add job and service

Files likely touched:

- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- `src/alpha_agent/cognition/loops/__init__.py`
- `tests/cognition/test_feedback_attribution.py`

Implementation:

- Add `FeedbackAttributionJob` carrying: `session_id`, `turn_id`,
  `turn_received_event_id`, `user_message_id`, `user_message_text`,
  the foreground-snapshotted prompt messages, recalled belief handles, and
  recall tool message ids. The snapshot is built in the foreground from the
  already-assembled answer messages; the background thread performs no session
  message reads.
- Add `RealtimeFeedbackAttributionService` modeled on
  `DirectCompactExtractionService`: bounded worker slots, daemon threads,
  `submit(job)`, `shutdown(wait=...)`, and audit records for
  `feedback_attribution_completed` / `feedback_attribution_failed` /
  `feedback_attribution_saturated`.
- `_run_job` composes `job.prompt_messages` plus one attribution instruction
  message listing recalled belief ids and short contents, calls the provider
  with `JSON_OBJECT_RESPONSE_FORMAT` and no tools, and validates output with
  the Task 2 contract.
- For each non-irrelevant verdict, emit `RECEIVED_FEEDBACK` with:
  - payload: `turn_id`, `session_id`, `feedback_kind` in
    `belief_confirmed` / `belief_contradicted` / `belief_corrected`,
    `matched_expected = (verdict == "confirmed")`, `belief_id`, `verdict`,
    `evidence_quote`, `user_message_id`, and recall tool message ids;
  - inputs referencing the belief and the user message;
  - `causal_parents` containing the turn-received event id.
- `submit(job)` acquires a worker slot before claiming ledger rows and
  starting the thread; saturated submissions write the saturation audit record
  and claim no rows.
- Mark ledger rows succeeded after event emission, and failed with the error
  on any exception. Task 7 moves the success point to after the Phase 3
  consequence writes once those exist. `RECEIVED_FEEDBACK` emission is
  at-least-once: a retry after partial failure may re-emit events, while all
  downstream consequences stay idempotent.
- LLM traces use the shared trace logger with job-identifying metadata.

Acceptance criteria:

- [ ] A scripted provider response produces validated verdicts and one
      `RECEIVED_FEEDBACK` event per non-irrelevant verdict.
- [ ] Irrelevant verdicts emit no events.
- [ ] Invalid provider output marks the ledger row failed, writes a failure
      audit record, and emits no events.
- [ ] Saturated submissions leave no claimed ledger rows.
- [ ] The service never writes session messages or runtime turn state.
- [ ] Shutdown waits for started jobs when requested.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
```

### Task 5: Add runtime submit hook

Files likely touched:

- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/daemon/manager.py`
- `src/alpha_agent/daemon/runtime.py`
- `tests/test_agent_loop.py`
- `tests/test_daemon_runtime.py`

Implementation:

- Add a `feedback_attribution_submitter` constructor parameter to
  `AlphaAgent`, alongside `compact_extraction_submitter`.
- In `respond()`, after the answer prompt messages are built (they already
  include the appended user message): run the Task 1 parser with the new user
  message ordinal; when it returns at least one belief, build the job from
  in-memory data and call the submitter.
- Record `feedback_attribution_submitted` and related ids in turn debug.
- Submitter exceptions and saturation are recorded via runtime trace or audit
  only; the turn proceeds unaffected.
- Construct `RealtimeFeedbackAttributionService` in the daemon runtime next to
  `DirectCompactExtractionService`, pass its `submit` through the agent
  factory, and shut it down on daemon stop.

Acceptance criteria:

- [ ] A turn following a recall-bearing turn submits exactly one job with the
      correct belief handles and user message.
- [ ] Turns without prior recall submit nothing.
- [ ] Submitter failure does not fail or delay the turn.
- [ ] Daemon start wires the service; daemon stop shuts it down.

Verification:

```bash
uv run pytest tests/test_agent_loop.py tests/test_daemon_runtime.py -q
```

## Phase 3: Feedback Consequences

### Task 6: Append feedback history for all non-irrelevant verdicts

Files likely touched:

- `src/alpha_agent/cognition/state_service.py`
- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- `tests/cognition/test_feedback_attribution.py`

Implementation:

- Add a state service write path that appends one `FeedbackEntry` to an active
  atomic belief's `feedback_history` and writes a
  `belief_feedback_recorded` audit record in the same transaction.
- Entry format is a compact JSON string with `at`, `kind` (`confirmed`,
  `contradicted`, or `corrected`), and the `RECEIVED_FEEDBACK` event id.
- Apply this inside the service job for every non-irrelevant verdict, so a
  challenged belief keeps its challenge entry even when conflict review later
  keeps the belief or stalls in pending confirmation.
- Throttle: skip the append when an entry of the same kind already exists for
  the same UTC date; the append is therefore idempotent under retries, and the
  event itself is still emitted.

Acceptance criteria:

- [ ] Confirmed, contradicted, and corrected verdicts each append exactly one
      feedback entry with the matching kind.
- [ ] A second same-kind verdict on the same UTC date emits an event but does
      not append a duplicate entry.
- [ ] Inactive target beliefs are skipped without error.
- [ ] Each append produces an audit record.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
```

### Task 7: Route contradictions into conflict review

Files likely touched:

- `src/alpha_agent/cognition/loops/feedback_attribution.py`
- `tests/cognition/test_feedback_attribution.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- For `contradicted` and `corrected` verdicts on still-active beliefs, create
  one `CONFLICT_REVIEW` source window per belief.
- Window `source_refs` is the single synthetic ref
  `BackgroundSourceRef("conflict", f"belief_feedback:{belief_id}:{user_message_id}")`;
  `create_source_window` requires a non-empty ref list, and embedding the
  belief/message pair in the source id keeps progress-row primary keys unique
  when the same belief is challenged again later.
- Window metadata must contain `active_belief_ids: [belief_id]`, which
  `MemoryConflictReviewWorker` reads to resolve valid update targets, plus
  `belief_id`, `belief_content`, `verdict`, `evidence_quote`,
  `feedback_event_id`, `session_id`, and `user_message_id`.
- Derive the window idempotency key deterministically from
  `(belief_id, user_message_id)`: the attribution contract guarantees one
  verdict per belief id for a given user message, and retries that re-emit
  `RECEIVED_FEEDBACK` events must map to the same window.
- Move the service's ledger success point so attribution rows are marked
  succeeded only after history appends (Task 6) and conflict windows are
  written.
- Confirm the metadata shape renders correctly through
  `_conflict_review_instruction_message` and is resolvable by
  `MemoryConflictReviewWorker` without changes to its decision logic.
- The user-verbatim `evidence_quote` is rendered into the conflict review
  prompt by design; safety against quoted injection-like text relies on the
  conflict review output contract (target whitelist, schema, authority
  ceiling) bounding any steered output, not on string blacklisting. Do not
  re-add injection rejection for this field downstream.
- Skip enqueueing when the belief is no longer active.

Acceptance criteria:

- [ ] Contradicted and corrected verdicts enqueue one pending conflict review
      window each.
- [ ] The existing conflict review worker consumes the window, resolves the
      target belief from `active_belief_ids`, and can produce supersede or
      pending-confirmation outcomes.
- [ ] Duplicate feedback events do not enqueue duplicate windows.
- [ ] Windows are not created for superseded or retracted beliefs.
- [ ] Attribution ledger rows are marked succeeded only after history appends
      and conflict windows are written.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_attribution.py -q
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

## Phase 4: End-to-End Verification

### Task 8: Add scripted feedback loop test

Files likely touched:

- `tests/cognition/test_feedback_loop_end_to_end.py` (new)

Implementation:

- Script the mock provider end to end:
  1. Seed an active, deliberately wrong preference belief.
  2. Turn 1: the model calls `memory_recall` and the seeded belief is
     returned.
  3. Turn 2: the user message contradicts the belief; the runtime submits the
     attribution job; run the job synchronously in the test.
  4. Assert one `RECEIVED_FEEDBACK` event with `belief_contradicted`, a
     verbatim evidence quote, and a `contradicted` feedback history entry on
     the seeded belief.
  5. Assert one pending `CONFLICT_REVIEW` window.
  6. Run `MemoryConflictReviewWorker` with a scripted supersede decision.
  7. Assert the seeded belief is superseded, the corrected belief is active,
     and a fresh recall returns the corrected belief.
- Assert ledger rows progress pending → claimed → succeeded across the flow.

Acceptance criteria:

- [ ] The full seeded-wrong-belief flow passes with deterministic mock
      scripting.
- [ ] Every step is traceable through cognitive events, ledger rows, and audit
      records.

Verification:

```bash
uv run pytest tests/cognition/test_feedback_loop_end_to_end.py -q
uv run pytest -q
uv run mypy src tests
```
