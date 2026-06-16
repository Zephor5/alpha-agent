# Consolidation Batch Reconciliation Plan

## Status

Ready for implementation.

## Goal

Refactor background memory consolidation so extraction remains a source-window
evidence boundary, consolidation becomes an atomic-memory reconciliation boundary,
and summary remains a target-level compression boundary.

## Non-Goals

- Do not preserve compatibility with existing database rows or old background windows.
- Do not make consolidation read raw session messages.
- Do not make consolidation read summary content.
- Do not change `memory_conflict_review` to the new batch decisions contract in this task.
- Do not tune summary thresholds in this task.
- Do not add compatibility flags for old single-operation consolidation behavior.

## Stage Boundaries

### Extraction

Extraction reads raw session/import/compact source messages and creates active
`BACKGROUND_EXTRACTED` atomic belief inputs.

Extraction must not decide final memory lifecycle, merge duplicate memories, or
write summary beliefs.

### Consolidation

Consolidation consumes active `BACKGROUND_EXTRACTED` atomic beliefs and produces
final atomic memory state.

Consolidation may:

- promote an extracted atomic belief in place to `BACKGROUND_CONSOLIDATED`;
- create a rewritten or synthesized `BACKGROUND_CONSOLIDATED` atomic belief;
- strengthen, supersede, retract, or archive one existing final atomic target;
- skip extracted inputs that should not become final memory.

Consolidation must not create summary beliefs, read summary content, or reread
raw transcript/source messages.

### Summary

Summary reads final active atomic memories and writes `BACKGROUND_SUMMARIZED`
summary beliefs for selected summary targets.

Summary must not read active `BACKGROUND_EXTRACTED` beliefs, reconcile atomic
belief lifecycle, or produce atomic belief operations.

## Execution Model

Keep the current background tick orchestration model.

Each background tick loops while any stage makes progress:

1. drain extraction;
2. drain consolidation;
3. drain conflict review;
4. drain summary;
5. drain archive.

Extraction is the only stage batched by session. It selects up to
`cognition.background.extraction.max_sessions_per_pass` eligible inactive
sessions per pass, and drains each selected session's extraction backlog before
returning to the next stage.

Consolidation is not session-scoped. It scans the current global set of active
`BACKGROUND_EXTRACTED` atomic beliefs, filters them by retryable consolidation
source progress, groups them into owner buckets, and consumes selected extracted
belief ids through consolidation source windows.

The handoff from extraction to consolidation is therefore:

- extraction writes active `BACKGROUND_EXTRACTED` atomic beliefs;
- consolidation treats those belief ids as `BackgroundSourceRef("atomic_belief", id)`;
- successful consolidation marks those source refs processed and removes them
  from the active extracted-source backlog by promotion or archival.

Do not add a session-local consolidation pipeline. Batch consolidation consumes
whatever extracted beliefs are currently visible when the consolidation stage
runs.

## Definitions

- `owner bucket`: the consolidation lane identified by `scope + about`.
- `final atomic memory`: an active atomic belief whose derivation stage is one of
  `BACKGROUND_CONSOLIDATED`, `TOOL_WRITTEN`, or `HUMAN_CONFIRMED`.
- `extracted source`: an active atomic belief whose derivation stage is
  `BACKGROUND_EXTRACTED`.
- `consumed source`: an extracted source listed in a consolidation decision.

## Confirmed Design Decisions

- Consolidation does not receive summary text.
- Summary reads all final atomic memories, not only background consolidated
  memories.
- Consolidation batches by `scope + about`; topic/domain only affect ordering and
  active context retrieval.
- Each LLM batch output contains `decisions[]`.
- Every extracted source in the batch must be consumed exactly once.
- `skip` archives consumed extracted sources.
- Isolated extracted sources with no reconciliation context are promoted
  deterministically without an LLM call.
- Deterministic promotion updates the existing belief in place from
  `BACKGROUND_EXTRACTED` to `BACKGROUND_CONSOLIDATED`.
- `promote` is also a valid LLM decision for batch consolidation.
- Promotion keeps original belief sources unchanged; audit and stage-run output
  record the consolidation action.
- Source progress is the primary duplicate-processing guard.
- Consolidation window idempotency uses stage, target unit, source refs, and
  contract version; active context ids are metadata only.
- Background tick ordering stays stage-drained: extraction by session batch,
  then global consolidation, then conflict review, summary, and archive.
- Deterministic promotion still creates a consolidation source window and stage
  run.
- Batch consolidation does not read raw extraction source messages.
- Active context retrieval is deterministic in the first implementation.
- Summary thresholds stay unchanged.
- Add only minimal consolidation size controls:
  `max_extracted_per_batch` and `max_active_context`.
- Each decision may operate on at most one existing target belief.
- Batch persistence is all-or-nothing in one transaction.
- Failed batches may be reselected; no frozen batch queue is required.
- `BackgroundStage.CONSOLIDATION` and `BackgroundStage.CONFLICT_REVIEW`
  validation paths are separated.

## Batch Decision Operation Contract

Ordinary consolidation batch output keeps the standard background LLM envelope
and uses one stage-specific operation:

```json
{
  "operation": "consolidate_atomic_beliefs",
  "authority": "background_synthesized",
  "rationale": "Why this batch-level reconciliation is appropriate.",
  "payload": {
    "decisions": []
  },
  "source_span_note": null
}
```

`payload.decisions` contains the per-source decisions. Each decision consumes one
or more source ids from the selected extracted-source batch. Across the whole
output, every selected source id must appear exactly once.

Every decision must include:

- `operation`;
- `source_atomic_belief_ids`;
- `rationale`.

Use `rationale` consistently for all decision types, including `skip`; do not add
a separate `reason` field.

| Operation | Source ids | Target id | Atomic belief input | Persistence effect |
| --- | --- | --- | --- | --- |
| `promote` | exactly 1 | forbidden | forbidden | Update that source belief in place to `BACKGROUND_CONSOLIDATED` |
| `skip` | 1..N | forbidden | forbidden | Archive consumed extracted sources |
| `create` | 1..N | forbidden | required | Write one new `BACKGROUND_CONSOLIDATED` belief and archive consumed sources |
| `strengthen` | 1..N | exactly 1 | forbidden | Reaffirm target and archive consumed sources |
| `supersede` | 1..N | exactly 1 | required | Supersede target with one new belief and archive consumed sources |
| `retract` | 1..N | exactly 1 | forbidden | Retract target and archive consumed sources |
| `archive` | 1..N | exactly 1 | forbidden | Archive target and archive consumed sources |

Additional validation requirements:

- the only ordinary consolidation top-level operation is
  `consolidate_atomic_beliefs`;
- `authority`, `rationale`, `payload`, and optional `source_span_note` remain
  top-level fields, matching the existing background LLM output envelope;
- `payload` must contain exactly `decisions` for ordinary consolidation;
- every `source_atomic_belief_ids` item must be in the current consolidation
  window's atomic belief source refs;
- every `target_belief_id` must be in the selected final active atomic context;
- each decision may operate on at most one existing target belief;
- `create` with exactly one source and identical final fields is invalid; use
  `promote`;
- `promote` cannot merge or rewrite multiple sources because it preserves the
  source belief id.

## Per-Decision Provenance

Batch source windows may contain multiple extracted sources and multiple
decisions. New or updated beliefs must not blindly attach every source ref from
the whole batch.

Persistence must distinguish:

- window/run provenance: every decision may reference the shared consolidation
  source window and stage run;
- decision source provenance: each written or reaffirmed target attaches only
  the `atomic_belief` source refs consumed by that specific decision.

Operation-specific provenance rules:

- `promote` keeps the promoted source belief's original extraction sources
  unchanged and records promotion through audit/window/run output.
- `create` and `supersede` attach the consumed source belief refs for that
  decision, plus the shared consolidation window/run refs.
- `strengthen` reaffirms the target using only the consumed source belief refs
  for that decision, plus the shared consolidation window/run refs.
- `skip`, `retract`, and `archive` do not create a new final belief from the
  consumed source refs; they still archive the consumed extracted sources and
  record the operation audit.

## Implementation Tasks

Each task below is an implementation waypoint, not a mandatory standalone
release boundary. A task may leave the branch temporarily unverifiable when it
changes shared contracts or call chains that are completed by a later task.
Prefer keeping tests focused as each layer lands, but the hard requirement is
that the final integrated implementation satisfies the completion checklist and
full validation commands.

### Task 1: Add Consolidation Configuration

Add `cognition.background.consolidation` config support.

Acceptance criteria:

- `BackgroundConsolidationConfig` exposes `max_extracted_per_batch = 8`.
- `BackgroundConsolidationConfig` exposes `max_active_context = 20`.
- config loading, serialization, validation, and `config.example.toml` include
  the new section.
- background worker config passes these values to `MemoryConsolidationWorker`.
- remove `cognition.background.consolidation` from removed config sections.
- add flat config keys, positive integer validation, env overrides, and config
  serialization entries for both settings.

Verification:

- `uv run pytest tests/test_config.py -q`
- `uv run mypy src tests`

### Task 2: Define Batch Decisions Schema

Define the ordinary consolidation batch output schema and validated payload
model.

Acceptance criteria:

- each decision includes `operation`, `source_atomic_belief_ids`, and
  `rationale`.
- supported decision operations and required fields follow the Batch Decision
  Operation Contract.
- every source id in the selected batch is consumed exactly once across all
  decisions.
- operation-specific source cardinality is enforced, especially `promote`
  requiring exactly one source.
- every consumed source id is one of the current window's atomic belief source
  refs.
- every target id is one of the selected active context belief ids.
- duplicate-copy `create` with a single source is rejected; the model must use
  `promote`.

Verification:

- focused validator tests for valid mixed decisions.
- focused validator tests for missing source coverage, duplicate source
  consumption, unknown source ids, unknown target ids, multi-target operations,
  and duplicate-copy `create`.
- `uv run mypy src tests`

### Task 3: Split Consolidation and Conflict Review Contracts

Separate ordinary consolidation validation from conflict review validation and
route ordinary consolidation to the batch decisions schema.

Acceptance criteria:

- `BackgroundStage.CONSOLIDATION` validates the new batch decisions contract.
- `BackgroundStage.CONFLICT_REVIEW` keeps the existing single-operation contract.
- conflict review tests continue to pass without schema broadening.
- no summary operations are accepted by either consolidation path.
- legacy single-operation `create`, `strengthen`, `supersede`, `retract`,
  `archive`, and `skip` outputs are rejected for ordinary consolidation but
  still accepted for conflict review where applicable.

Verification:

- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `uv run mypy src tests`

### Task 4: Add In-Place Promotion Persistence

Add persistence support for changing an extracted atomic belief into a final
consolidated atomic belief without creating a new belief id.

Acceptance criteria:

- belief projection/state service can promote an active atomic belief by
  rebuilding an `AtomicBelief` record and using the normal atomic upsert path.
- promotion must keep the serialized `record`, table columns, indexes, and FTS
  data consistent; a column-only SQL update is not acceptable.
- promotion only applies to active `BACKGROUND_EXTRACTED` atomic beliefs.
- promotion preserves topic, content, scope, about, validity, update policy, and
  original sources.
- promotion writes an audit record with operation `promote`.
- promotion emits the same belief id as the consolidation stage-run output ref.
- promotion marks source refs, source window, and stage run processed.

Verification:

- focused state service test for successful promotion.
- focused state service test rejecting promotion of non-extracted or inactive
  beliefs.

### Task 5: Apply Batch Decisions Atomically

Update `accept_background_llm_json` persistence for ordinary consolidation
decisions.

Acceptance criteria:

- all decisions in a batch are applied in one transaction.
- `promote` updates source belief stage in place.
- `skip` archives consumed extracted sources and writes no final belief.
- `create` writes a new `BACKGROUND_CONSOLIDATED` atomic belief, attaches only
  that decision's consumed source refs, and archives consumed extracted sources.
- `strengthen` reaffirms the target using only that decision's consumed source
  refs and archives consumed extracted sources.
- `supersede` writes a replacement target and archives consumed extracted
  sources.
- `retract` and `archive` update the target lifecycle and archive consumed
  extracted sources.
- no consumed source remains active `BACKGROUND_EXTRACTED` after a successful
  batch.
- failed validation or persistence leaves the batch unapplied.

Verification:

- focused state service tests for each decision operation.
- focused persistence tests proving `create`, `strengthen`, and `supersede`
  attach only per-decision source refs, not every source ref in the batch window.
- transaction rollback test for one invalid decision in a mixed batch.

### Task 6: Implement Deterministic Active Context Retrieval

Select active final atomic context for LLM batch consolidation without semantic
search.

Acceptance criteria:

- retrieval is limited to the same owner bucket.
- eligible context excludes `BACKGROUND_EXTRACTED` and summary beliefs.
- eligible context includes active final atomic memories.
- ranking prefers normalized content exact match, topic match, token overlap,
  memory kind match, shared target domains, and recency.
- context size is capped by `max_active_context`.
- retrieval order is stable.

Verification:

- focused retrieval tests for eligibility, ranking, cap, and stable ordering.

### Task 7: Select Batch Consolidation Candidates

Change `MemoryConsolidationWorker` candidate selection from one extracted belief
to one owner-bucket batch.

Acceptance criteria:

- candidates are grouped by `scope + about`.
- candidate batch size is capped by `max_extracted_per_batch`.
- active or claimed consolidation windows prevent concurrent processing of the
  same target unit.
- failed source progress remains retryable.
- selected source refs are active `BACKGROUND_EXTRACTED` atomic beliefs.
- candidate selection uses the deterministic active context retrieval from Task 6.
- candidate metadata records source belief ids, active context ids, selection
  reason, owner bucket, and consolidation contract version.
- window idempotency key includes stage, target unit, selected source refs, and
  consolidation contract version.
- active context ids are not part of the idempotency key.

Verification:

- worker selection tests for multi-source same-bucket batches.
- worker selection tests for retryable failed sources.
- worker selection tests that active windows block duplicate processing.

### Task 8: Add Deterministic Promotion Fast Path

Skip LLM calls for isolated extracted sources with no reconciliation context.

Acceptance criteria:

- deterministic promotion uses the same final active atomic context lookup as LLM
  batch consolidation.
- deterministic promotion runs only when the owner bucket has exactly one active
  extracted source and no final active atomic context.
- deterministic promotion does not require an LLM provider; missing provider is
  only an error for candidates that require LLM batch consolidation.
- deterministic promotion creates and processes a consolidation source window.
- deterministic promotion starts and finishes a consolidation stage run.
- metadata records `operation = promote` and `llm_called = false`.
- no LLM provider call is made for deterministic promotion.
- non-isolated extracted sources still use LLM batch consolidation.

Verification:

- worker test asserting isolated source is promoted without provider calls.
- worker test asserting source plus active context calls the provider.
- worker test asserting two extracted sources in the same bucket call the
  provider.

### Task 9: Update Consolidation Prompt

Update ordinary consolidation prompt to reflect batch reconciliation.

Acceptance criteria:

- prompt says consolidation reconciles atomic memory inputs, not summaries.
- prompt states it cannot see raw source messages.
- prompt states every supplied extracted source must be consumed exactly once.
- prompt explains `promote` versus `create`.
- prompt says `create` is for rewritten/synthesized final atomic beliefs, not
  copying one source unchanged.
- prompt keeps same-language guidance for output topic/content.
- prompt material includes selected extracted sources and selected final active
  atomic context only.
- prompt shows the standard output envelope with
  `operation = "consolidate_atomic_beliefs"` and decisions under
  `payload.decisions`.

Verification:

- prompt-focused tests assert required boundary language is present.
- no prompt tests lock full prompt text.

### Task 10: Expand Summary Eligibility to Final Atomic Memories

Change summary source eligibility from only `BACKGROUND_CONSOLIDATED` to all
final active atomic memories.

Acceptance criteria:

- eligible summary source stages are `BACKGROUND_CONSOLIDATED`, `TOOL_WRITTEN`,
  and `HUMAN_CONFIRMED`.
- `BACKGROUND_EXTRACTED` is excluded.
- `BACKGROUND_SUMMARIZED` is excluded.
- summary target grouping behavior remains target-based.
- summary thresholds remain unchanged.

Verification:

- summary worker tests for included and excluded derivation stages.
- existing summary target tests continue to pass.

### Task 11: Update Background Service Backlog Counting

Align consolidation backlog counting with batch selection and promotion.

Acceptance criteria:

- pending consolidation count reflects active retryable `BACKGROUND_EXTRACTED`
  sources.
- deterministic promotion completion removes the source from future pending
  counts.
- LLM batch completion removes every consumed source from future pending counts.
- failed windows leave sources retryable.

Verification:

- background service drain tests for promotion, LLM batch, and failed retry
  behavior.

### Task 12: End-to-End Verification

Add or update end-to-end tests for the new pipeline shape.

Acceptance criteria:

- imported sessions can extract beliefs, promote isolated ones without LLM
  consolidation duplication, and make them eligible for summary.
- multiple extracted beliefs in the same owner bucket are sent in one LLM batch.
- a mixed decision batch closes every source exactly once.
- a mixed decision batch records per-decision provenance, not whole-window
  provenance on every written belief.
- no active `BACKGROUND_EXTRACTED` source remains after successful consolidation.
- conflict review still uses the old single-operation path.

Verification:

- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`

## Implementation Order

1. Add config and worker config plumbing.
2. Add batch decisions validation.
3. Split ordinary consolidation and conflict review validation paths.
4. Add in-place promotion persistence.
5. Add batch decision persistence.
6. Add deterministic active context retrieval.
7. Refactor consolidation candidate selection.
8. Add deterministic promotion fast path.
9. Update consolidation prompt.
10. Expand summary eligibility.
11. Update backlog counting and service drain tests.
12. Run full validation.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Batch validator becomes too permissive | Consolidation and summary boundaries blur | Separate consolidation and conflict review validators; reject summary operations in consolidation |
| Promotion loses provenance | Final memory cannot be traced to source evidence | Preserve original belief sources and record promotion audit/stage-run output |
| Promotion corrupts projection state | Record JSON, table columns, and search indexes diverge | Promote through reconstructed `AtomicBelief` upsert, not direct column-only SQL |
| Batch provenance over-attaches sources | Final beliefs cite unrelated extracted inputs from the same window | Build write/reaffirm sources from each decision's consumed source ids |
| Batch decisions partially apply | Source coverage and progress become inconsistent | Apply the whole batch in one transaction |
| Provider gate blocks deterministic promotion | Isolated sources cannot promote when no LLM provider is configured | Select candidate and detect deterministic promotion before requiring provider |
| Context retrieval is unstable | Prompt prefix churn and flaky tests | Use deterministic ranking and stable tie-breakers |
| New summary eligibility admits extracted noise | Summary can summarize unfinalized memories | Explicitly include only final atomic stages and exclude `BACKGROUND_EXTRACTED` |

## Completion Checklist

- [ ] Config supports consolidation batch and active context caps.
- [ ] Ordinary consolidation uses batch decisions.
- [ ] Conflict review keeps the single-operation contract.
- [ ] Isolated extracted beliefs promote without an LLM call.
- [ ] Isolated extracted beliefs promote even when no LLM provider is configured.
- [ ] LLM consolidation supports `promote` and rejects duplicate-copy `create`.
- [ ] Batch-created or reaffirmed beliefs attach only per-decision source refs.
- [ ] Consumed extracted sources close exactly once.
- [ ] Summary reads final atomic memories and excludes extracted memories.
- [ ] Full validation passes.
