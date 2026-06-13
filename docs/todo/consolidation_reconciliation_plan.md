# Consolidation Reconciliation Plan

## Objective

Upgrade background memory consolidation from single-draft handling to
bucket-based reconciliation over pending extracted drafts and related active
beliefs.

## Target Behavior

- Consolidation selects one coherent reconciliation bucket per run.
- A bucket contains pending extracted drafts that share the same reconciliation
  target and the active beliefs related to those drafts.
- Related active beliefs are retrieved through deterministic structural,
  lexical, recency, and summary-guided strategies.
- The LLM returns a single consolidation plan with one or more decisions.
- Every pending source draft in the bucket is consumed by exactly one decision.
- Persistence validates the whole plan before applying it in one transaction.
- Source draft progress, source window progress, stage run output refs, and
  audit records are written consistently for every accepted plan.

## Phase 1: Define Reconciliation Contracts

### Task 1: Add plan output schema

Files likely touched:

- `src/alpha_agent/cognition/background_llm_contract.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Add a `consolidation_plan` operation for consolidation-stage LLM output.
- Add `payload.decisions` as a non-empty array.
- Define decision variants for `create`, `strengthen`, `supersede`, `retract`,
  `archive`, `drop_as_noise`, and `pending-confirmation`.
- Require each decision to include `source_draft_ids`.
- Require update-like decisions to include validated `target_belief_ids`.
- Require create-like decisions to include one id-less `atomic_belief_draft`.

Acceptance criteria:

- [ ] The schema accepts a valid multi-decision consolidation plan.
- [ ] The schema rejects a plan with no decisions.
- [ ] The schema rejects decisions without `source_draft_ids`.
- [ ] The schema rejects generated belief ids, source refs, provenance,
      idempotency keys, confidence, scores, and numeric strength fields inside
      plan decisions.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

### Task 2: Add deterministic plan validation

Files likely touched:

- `src/alpha_agent/cognition/background_llm_contract.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Validate every `source_draft_id` against the current source window input refs.
- Validate that every source draft appears in exactly one decision.
- Validate that every target belief id is in `allowed_target_belief_ids`.
- Validate that every target belief is active and atomic before persistence.
- Validate that create-like and supersede-like drafts use allowed about refs.
- Validate that each decision operation has exactly the payload keys required
  for that operation.
- Return typed validated plan and decision payload objects.

Acceptance criteria:

- [ ] Duplicate source draft consumption is rejected.
- [ ] Missing source draft consumption is rejected.
- [ ] Unknown source draft ids are rejected.
- [ ] Unknown target belief ids are rejected.
- [ ] Per-operation payload shape errors include the decision index.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
uv run mypy src tests
```

## Phase 2: Build Related Active Belief Retrieval

### Task 3: Add consolidation context retriever

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/search_tokenizer.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Add a `ConsolidationContextRetriever` or equivalent module-local helper.
- Group pending `BACKGROUND_EXTRACTED` drafts by reconciliation target.
- Build deterministic related active belief candidates for each target.
- Return source drafts, related active beliefs, allowed target ids, allowed
  about refs, retrieval reasons, and prompt metadata.
- Use stable ordering for all returned collections.

Acceptance criteria:

- [ ] Pending extracted drafts are grouped by scope/about target.
- [ ] Related active beliefs are returned with deterministic ordering.
- [ ] Retrieval metadata records why each active belief was included.
- [ ] Repeated retrieval over unchanged state returns the same bucket.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

### Task 4: Add structural retrieval strategies

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Include active beliefs with the same exact scope/about target.
- Include active beliefs with the same normalized object when present.
- Include active beliefs with the same project descriptor when present.
- Include active beliefs directly referenced by source/provenance fields when
  available in current records.
- Include active beliefs linked to the same session, counterpart, subject, or
  project reference.

Acceptance criteria:

- [ ] Same scope/about active beliefs are always included.
- [ ] Same object active beliefs are included.
- [ ] Same project descriptor active beliefs are included.
- [ ] Directly linked active beliefs are included.
- [ ] Structural candidates appear before weaker retrieval candidates.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

### Task 5: Add lexical retrieval strategies

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/search_tokenizer.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Tokenize draft content, object, and project descriptor.
- Tokenize active belief content, object, and project descriptor.
- Score active beliefs by token overlap.
- Include high-scoring lexical matches in the related active belief set.
- Store lexical score and matched tokens in retrieval metadata.

Acceptance criteria:

- [ ] Lexically similar active beliefs are included even when scope/about is
      broader than the exact target.
- [ ] Ranking applies stop-word filtering or weighting before lexical matches
      are capped.
- [ ] CJK and technical tokens are handled through the project tokenizer.
- [ ] Lexical retrieval output is deterministic.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

### Task 6: Add recency and summary-guided retrieval strategies

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Include recently changed active beliefs for the same session, counterpart,
  subject, or project target.
- Include active beliefs recently strengthened, superseded, retracted, or
  archived when their target matches the pending drafts.
- Include active summary beliefs for the same summary target.
- Include active atomic beliefs referenced by matching summary source ids.
- Store recency and summary retrieval reasons in metadata.

Acceptance criteria:

- [ ] Recent same-target beliefs are included.
- [ ] Summary source beliefs are included when a matching summary exists.
- [ ] Retrieval reasons distinguish recency matches from summary-guided
      matches.
- [ ] Retrieval caps preserve all structural candidates before lower-priority
      candidates.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

## Phase 3: Migrate Consolidation Worker

### Task 7: Select bucket candidates instead of single drafts

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `src/alpha_agent/cognition/loops/background_service.py`
- `src/alpha_agent/config.py`
- `config.example.toml`
- `tests/test_config.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Replace single-draft candidate selection with bucket candidate selection.
- Rename consolidation sizing config to resource-oriented plan limits.
- Add config for maximum drafts per plan.
- Add config for maximum related active beliefs per plan.
- Add config for maximum lexical candidates per draft.
- Keep pending consolidation count based on unprocessed extracted drafts.
- Store selected draft ids, related active belief ids, and retrieval metadata on
  the source window.

Acceptance criteria:

- [ ] One worker run can select multiple pending extracted drafts in the same
      bucket.
- [ ] Source window metadata records draft ids and related active belief ids.
- [ ] Config loading, persistent config display, and example config expose the
      new plan limits.
- [ ] Pending consolidation count still reflects unprocessed extracted drafts.

Verification:

```bash
uv run pytest tests/test_config.py tests/cognition/test_consolidation_loop.py -q
uv run mypy src tests
```

### Task 8: Update consolidation prompt

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Change the instruction from one decision to one consolidation plan.
- Include source drafts as an array with ids and memory fields.
- Include related active beliefs as an array with ids, memory fields, and
  retrieval reasons.
- Include allowed target ids and allowed about refs.
- Require every source draft id to be consumed by the plan.
- Require one top-level JSON object matching the plan schema.

Acceptance criteria:

- [ ] The prompt includes `payload.decisions`.
- [ ] The prompt includes every selected source draft id.
- [ ] The prompt includes related active belief ids and retrieval reasons.
- [ ] The prompt requires every source draft id to be consumed.
- [ ] The prompt requires exactly one top-level JSON object.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

## Phase 4: Apply Plans Transactionally

### Task 9: Persist validated consolidation plans

Files likely touched:

- `src/alpha_agent/cognition/state_service.py`
- `src/alpha_agent/cognition/background_llm_contract.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Add `_apply_consolidation_plan_output`.
- Validate active target lifecycle inside the transaction.
- Apply each decision in deterministic order.
- Create consolidated beliefs for create-like decisions.
- Reaffirm targets for strengthen decisions.
- Supersede targets for supersede decisions.
- Mark target lifecycle for retract and archive decisions.
- Write confirmation records for pending-confirmation decisions.
- Record drop-as-noise decisions through audit and source progress.
- Collect output refs for all written or updated beliefs.

Acceptance criteria:

- [ ] A valid plan applies all decisions in one transaction.
- [ ] A failed decision rolls back the whole plan.
- [ ] Output refs include all written or updated belief refs.
- [ ] Drop-as-noise decisions consume source drafts through audit-only records.
- [ ] Confirmation-required decisions write auditable pending records.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
uv run mypy src tests
```

### Task 10: Archive consumed extracted drafts

Files likely touched:

- `src/alpha_agent/cognition/state_service.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Archive every consumed `BACKGROUND_EXTRACTED` source draft after plan
  application.
- Mark each consumed source draft processed for consolidation.
- Mark the source window processed only after all source refs are processed.
- Finish the stage run with complete output refs.

Acceptance criteria:

- [ ] Every consumed source draft is archived after successful plan application.
- [ ] Every consumed source draft has consolidation progress marked processed.
- [ ] The source window is processed after successful plan application.
- [ ] The stage run succeeds with complete output refs.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

## Phase 5: Tighten Tests and Observability

### Task 11: Add end-to-end consolidation plan tests

Files likely touched:

- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Add tests for a plan that creates one new belief and strengthens one existing
  belief.
- Add tests for a plan that supersedes one active belief.
- Add tests for a plan that drops one source draft as noise.
- Add tests for a plan that routes one source draft to pending confirmation.
- Add tests for transactional rollback on invalid target lifecycle.

Acceptance criteria:

- [ ] Multi-decision plan writes expected beliefs and source progress.
- [ ] Supersede plan archives the target and writes the replacement.
- [ ] Drop-as-noise plan records audit-only source draft consumption.
- [ ] Pending confirmation plan writes auditable confirmation state.
- [ ] Invalid mixed plan rolls back all decisions.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

### Task 12: Add retrieval and trace observability

Files likely touched:

- `src/alpha_agent/cognition/loops/workers/_common.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `tests/cognition/test_consolidation_loop.py`

Implementation:

- Add retrieval metadata to background LLM trace metadata.
- Include bucket id, source draft count, related active belief count, and
  retrieval reason counts.
- Include plan decision count after validation succeeds.
- Include decision operation counts in the stage run metadata or audit payload.

Acceptance criteria:

- [ ] LLM traces show the consolidation bucket and retrieval counts.
- [ ] Audit records show plan decision operation counts.
- [ ] Stage run metadata can be used to inspect the plan scope.

Verification:

```bash
uv run pytest tests/cognition/test_consolidation_loop.py -q
```

## Final Checkpoint

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src tests`
- [ ] `uv run pytest -q`
- [ ] Active background drain config keys are documented in
      `config.example.toml`; no `cognition.background.consolidation` keys
      are expected.
- [ ] Existing extraction and summary worker tests still pass.
