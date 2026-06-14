# Cognition Belief Contract Refactor Plan

Status: complete; full validation gate passed
Date: 2026-06-14

## Goal

Tighten the cognition belief contract so background extraction, memory proposal,
projection, and recall use clear field responsibilities:

- `topic` is the short retrieval/consolidation key.
- `content` is the full natural-language assertion.
- `about` owns scope and target references.
- Entity lookup only uses explicit entity references at both index-write time
  and query time.
- Imported conversation extraction avoids durable memory noise from historical
  assistant answers and single-turn technical questions.

Existing local database compatibility is not a constraint. Prefer direct refactor
over compatibility shims.

## Confirmed Decisions

- Rename belief `object` to `topic` for both atomic and summary beliefs.
- `topic` is required, non-empty, `maxLength = 64`, and must not equal `content`
  after trimming.
- `topic` is not an entity id and must never feed `belief_entity_index`.
- Delete `AtomicBelief.structure`.
- Keep `SummaryBelief.structure`, but restrict it to summary/domain-specific
  structured output.
- Delete `action_orientation` from both atomic and summary beliefs.
- Do not add an entity table in this change.
- Keep `belief_entity_index`, but only index existing explicit
  `Reference(kind="entity", id=...)` references.
- Entity-exact recall must only use explicitly supplied entity query inputs,
  not general `query`, `keywords`, `topic`, `content`, or non-entity `about`
  text.
- Keep `about` capable of containing `Reference(kind="entity")`, but normal
  extraction/consolidation prompts must not offer entity refs as allowed refs
  until a dedicated entity-resolution mechanism exists.
- Keep `update_policy.target_domain` and `update_policy.target_domains` on
  atomic beliefs as internal summary scheduling hints.
- Do not allow ordinary extraction/consolidation prompts to generate
  `target_domain(s)` by default.
- Ordinary extraction/consolidation validators must reject LLM-generated
  `update_policy.target_domain` and `update_policy.target_domains` unless a
  program-owned internal path explicitly permits summary scheduling hints.
- Add required `memory.topic` to the `memory_propose` tool schema.
- Keep `target_hint`, but only as an operation target-selection hint; it must not
  generate or overwrite belief `topic`.
- Remove program fallback from `target_hint`, `evidence`, or `content` to
  `topic`. Missing or invalid `topic` fails fast.
- `memory_propose` event/audit payloads must preserve explicit `topic`; they
  must not reintroduce an implicit object/topic through derived payload fields.

## Prompt And Memory Policy

- Both normal and import extraction prompts must define scope boundaries.
- `scope=self` is only for the current Alpha Agent's stable identity,
  capabilities, constraints, and behavior commitments.
- User-subject assertions, including content starting with `The user...` or
  `User...`, must not use `scope=self`.
- `scope=global` must not contain user profile assertions.
- User interests, preferences, work context, and historical interactions belong
  under `scope=counterpart` or should be skipped.
- Import extraction defaults to no `scope=global`.
- Import extraction defaults to no `scope=self`.
- Imported assistant output is context for interpreting a transcript, not durable
  knowledge by default.
- Single-turn technical explanations and generic imported assistant answers
  should normally produce no belief.
- Single-turn inferred interests should normally be skipped, not written as
  pending memory.
- Import counterpart memory may be active only when the user directly stated a
  stable fact or preference.
- Import counterpart memory should be pending or skipped when it is inferred from
  a single question, inferred capability, historical temporary state, or
  assistant answer.
- Ordinary non-import extraction may produce `scope=self`, but only narrowly;
  default to `scope=counterpart` for relationship-specific service preferences.
- `scope=self` from background extraction defaults to pending unless the source
  is trusted.
- Consolidation must not turn uncertain imported drafts into active beliefs via
  `create` or `supersede` with `requires_confirmation=false`.
- Import uncertainty must be enforced at validation or persistence boundaries,
  not only by prompt wording. If an imported draft is inferred from assistant
  output, a single question, inferred capability, or historical temporary state,
  it must remain pending or be skipped even if the model emits
  `requires_confirmation=false`.

## Output Shape Rules

- Atomic and summary draft schemas both require `topic`.
- Atomic drafts no longer accept `structure`.
- Summary drafts keep `structure` only where the summary kind/domain requires it.
- Domain summaries still require `structure.target_domain` to match the selected
  target domain.
- Ordinary summaries should not emit empty or meaningless `structure`.
- Each extracted belief should contain exactly one atomic assertion in `content`.
- `topic` should be described in prompts as a short topic phrase, not a sentence
  and not a full assertion.
- `topic == content` fails validation for background LLM output and
  `memory_propose`.
- Prompt examples should include negative cases for:
  - sentence-like `topic`
  - multi-claim `content`
  - `scope=self` with user-subject content
  - `scope=global` with user profile content
  - imported assistant answer becoming global knowledge
  - imported assistant identity becoming current Alpha self memory

## Implementation Tasks

### Task 1: Update Belief Model And Schema

Description: Apply the core data contract changes to models, SQLite schema, and
projection row serialization.

Acceptance criteria:
- `AtomicBelief` and `SummaryBelief` expose `topic`, not `object`.
- Belief model construction or the shared state write boundary rejects invalid
  `topic` values before storage.
- `AtomicBelief` has no `structure`.
- `SummaryBelief.structure` remains.
- Neither belief model has `action_orientation`.
- `state/schema.sql` and projection-local schema match the new fields.
- No compatibility reader for old `object`, atomic `structure`, or
  `action_orientation` remains.

Likely files:
- `src/alpha_agent/cognition/models/belief.py`
- `src/alpha_agent/state/schema.sql`
- `src/alpha_agent/cognition/projections/belief.py`
- cognition model/projection tests

### Task 2: Enforce Topic Validation

Description: Make `topic` a required, explicit field at every belief creation
boundary.

Acceptance criteria:
- Background atomic and summary draft schemas require `topic`.
- `topic` validates as non-empty, max 64 characters, and not equal to `content`
  after trimming.
- Invalid `topic` rejects the whole background LLM output.
- Direct state-service and tool-created beliefs use the same `topic`
  validation as background LLM output.
- No code path uses `content` truncation, `evidence`, or `target_hint` as a
  semantic fallback topic.
- Ordinary background LLM outputs reject `update_policy.target_domain` and
  `update_policy.target_domains` unless the call context is explicitly
  program-owned for summary scheduling.

Likely files:
- `src/alpha_agent/cognition/background_llm_contract.py`
- `src/alpha_agent/cognition/state_service.py`
- tests for background LLM validation and consolidation/extraction loops

### Task 3: Update Memory Propose Contract

Description: Make `memory_propose` provide explicit belief topics and clarify
tool field responsibilities.

Acceptance criteria:
- `memory_propose.memory.topic` is required.
- Tool schema descriptions exist for `topic`, `content`, `evidence`, `scope`,
  `target_hint`, and `reason`.
- `target_hint` is only used for target selection and is never written into
  `topic`.
- `_belief_object`-style fallback is removed or replaced by direct validated
  `topic` use.
- Parsed memory records, emitted `MEMORY_PROPOSED` payloads, state audit payloads,
  and accepted/pending belief writes all carry the explicit `topic`.
- `payload_contract` coverage is updated if any consumed event field changes.

Likely files:
- `src/alpha_agent/tools/memory_propose.py`
- `tests/cognition/test_memory_propose_tool.py`
- runtime prompt/tool tests where tool schema is asserted

### Task 4: Update Projection, Recall, And Entity Indexing

Description: Align search and indexing with `topic` and remove accidental entity
generation.

Acceptance criteria:
- Projection tables and FTS columns use `topic`.
- Recall reasons are renamed from `object_exact` / `object_partial` to
  `topic_exact` / `topic_partial`.
- Recall scoring uses the new reason names.
- `belief_entity_index` is populated only from explicit
  `Reference(kind="entity")` refs.
- `topic`, `content`, and non-entity `about` refs do not create entity ids.
- Entity-exact candidate collection only probes `BeliefSearchParams.entities`;
  `query` and `keywords` never trigger entity-index matching.
- FTS still indexes `content`, `topic`, and `about` search terms as text.

Likely files:
- `src/alpha_agent/cognition/projections/belief.py`
- `src/alpha_agent/tools/memory_recall.py`
- recall/projection tests

### Task 5: Update Extraction And Consolidation Prompts

Description: Encode the confirmed scope, import-noise, confirmation, and output
shape policy in prompts.

Acceptance criteria:
- Normal extraction prompt defines `self`, `counterpart`, and `global` scope
  boundaries.
- Import extraction prompt forbids default `self` and `global` extraction.
- Import extraction prompt treats historical assistant output as context, not
  durable knowledge by default.
- Import extraction prompt says single-turn inferred interests and one-off
  technical Q&A should normally be skipped.
- Prompts require `topic` as a short topic phrase and include negative examples.
- Consolidation prompt keeps uncertain imported drafts pending instead of
  auto-activating them.
- Consolidation allowed-about refs exclude `Reference(kind="entity", ...)` for
  ordinary background prompts until a dedicated entity-resolution mechanism
  exists.

Likely files:
- `src/alpha_agent/cognition/loops/workers/memory_extraction.py`
- `src/alpha_agent/cognition/loops/workers/memory_consolidation.py`
- `tests/cognition/test_consolidation_loop.py`
- extraction/import-focused tests

### Task 6: Preserve Summary Domain Guidance With Narrow Structure

Description: Keep summary-specific structured output while removing atomic
structure.

Acceptance criteria:
- Domain summary still requires `structure.target_domain`.
- `summary_target_domain()` continues to work for summary beliefs.
- Atomic summary scheduling reads only `update_policy.target_domain(s)`.
- `_target_domains_for_belief()` no longer reads atomic `structure`.
- Summary worker target derivation for ordinary scoped summaries uses scope-owner
  refs only; incidental entity refs in `about` must not become prompt-allowed
  refs or summary target keys.
- Tests covering memory-propose domain guidance still pass.

Likely files:
- `src/alpha_agent/cognition/domain_guidance.py`
- `src/alpha_agent/cognition/loops/workers/memory_summary.py`
- `tests/cognition/test_domain_summary_worker.py`
- `tests/cognition/test_memory_propose_tool.py`

### Task 7: Update Tests, Fixtures, And Docs

Description: Bring tests and active documentation in line with the new contract.

Acceptance criteria:
- No cognition tests construct beliefs with `object`, atomic `structure`, or
  `action_orientation`.
- Tests cover `topic == content` fail-fast behavior.
- Tests cover import extraction not creating self identity from historical
  assistant output.
- Tests cover import extraction not creating global knowledge from one-off
  assistant technical answers.
- Tests cover `The user...` content not being accepted as `scope=self`.
- Active docs mention `topic`, not `object`.

Likely files:
- `tests/cognition/`
- `tests/test_agent_loop.py`
- `docs/cognition/cognition.md`
- this plan file

Cleanup status on 2026-06-14:

- Remaining old-name scan hits are intentional legacy-rejection assertions,
  JSON Schema `type: object` declarations, summary-only `structure` fixtures,
  or this plan's historical decision text.
- Touched active docs use `topic` for the belief retrieval/consolidation key.

## Verification

Project validation gate completed on 2026-06-14:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

Result:

- `uv run ruff check .` passed.
- `uv run mypy src tests` passed.
- `uv run pytest -q` passed with 751 tests.

Targeted tests to run while iterating:

```bash
uv run pytest tests/cognition/test_belief_projection_apply.py -q
uv run pytest tests/cognition/test_memory_recall_tool.py -q
uv run pytest tests/cognition/test_memory_propose_tool.py -q
uv run pytest tests/cognition/test_consolidation_loop.py -q
uv run pytest tests/cognition/test_domain_summary_worker.py -q
```

## Non-Goals

- Do not introduce an entity table or entity projection.
- Do not implement old database compatibility.
- Do not add programmatic semantic topic generation.
- Do not solve event sourcing gaps in this refactor.
- Do not introduce quoted-evidence requirements.
- Do not redesign the whole cognition architecture beyond this belief contract
  cleanup.

## Open Questions

No blocking product/design questions remain for this scoped implementation.
Implementation may still uncover mechanical test or naming cleanup questions.
