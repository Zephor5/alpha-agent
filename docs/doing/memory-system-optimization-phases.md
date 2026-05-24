# Memory System Optimization Phases

Status: active planning
Date: 2026-05-24

## Objective

Improve Alpha Agent's memory system from a transparent MVP into a controlled,
auditable, multi-scope memory runtime. The priority is not to fully implement
every layer from `docs/memory_design.md`; the priority is to make memory
capture, storage, retrieval, correction, and prompt injection trustworthy.

Current implementation strengths:

- `conversation_messages` is append-only and can replay session context.
- `session_context_states` provides compressed short-term session projection.
- episodic, semantic, and procedural memories exist as separate layers.
- retrieval is explicit and inspectable, with FTS/LIKE fallback and scoring.
- memory review exists as a CLI preview/approval path.
- runtime traces expose important memory and LLM events.

Current implementation risks:

- long-term memories do not have a first-class user/channel/project scope.
- extracted memories can be persisted immediately without a candidate lifecycle.
- semantic memory has no active/superseded/deleted/rejected status model.
- duplicate and contradictory memories are not resolved beyond exact triple
  upsert.
- extraction is deterministic and English-pattern heavy.
- consolidation is manual and limited to re-extracting semantic facts from
  episodic memories.
- retrieved memory enters the prompt without enough source/status context.

## Guiding Decisions

- Build a memory controller before adding richer memory layers.
- Treat memory scope and lifecycle as foundational data model concepts.
- Keep append-only transcript messages as the source of truth.
- Use natural-language atomic memories for LLM consumption, with weak structured
  fields for search, conflict checks, and audit.
- Keep vector retrieval optional and later; it should be one retriever among
  several, not the owner of memory semantics.
- Do not build scene, persona, or graph aggregation until atomic memory quality
  and correction flows are reliable.
- Prefer direct schema/runtime refactors over compatibility shims.

## Phase 0: Baseline And Evaluation

Goal: make memory behavior measurable before changing storage semantics.

Tasks:

- [ ] Add a small memory behavior fixture set covering preferences, facts,
  corrections, project state, procedure hints, and "do not remember" cases.
- [ ] Add retrieval evaluation helpers that report retrieved ids, scores, score
  components, and whether expected memories were selected.
- [ ] Add extraction evaluation helpers that compare candidates against expected
  type, content, scope, confidence, and source ids.
- [ ] Add prompt inspection tests for memory source/status rendering.

Acceptance criteria:

- [ ] `uv run pytest tests/test_memory_store.py tests/test_retrieval.py
  tests/test_prompt_builder.py tests/test_consolidation.py -q` passes.
- [ ] A failing retrieval, extraction, or prompt-injection regression can be
  explained from test output without manually inspecting SQLite.
- [ ] The evaluation fixtures do not depend on external LLM calls.

Likely files:

- `tests/test_retrieval.py`
- `tests/test_memory_store.py`
- `tests/test_prompt_builder.py`
- `tests/test_consolidation.py`
- new test helper under `tests/` if needed

## Phase 1: Memory Scope And Controller

Goal: route all memory reads/writes through one explicit policy layer and stop
global memory pollution before adding richer behavior.

Tasks:

- [ ] Add a `MemoryScope` domain model representing at least global user,
  platform user, chat/thread session, and project scopes.
- [ ] Add scope columns or a normalized scope table for episodic, semantic,
  procedural, candidate, and access-log records.
- [ ] Derive memory scope from CLI and gateway `source_metadata`.
- [ ] Introduce a `MemoryController` or `MemoryPipeline` that owns capture,
  extraction, candidate persistence, promotion, retrieval policy, and trace
  emission.
- [ ] Keep `AlphaAgent.respond()` as the readable orchestration path, but move
  memory policy decisions out of the runtime method body.
- [ ] Update retrieval to filter by allowed scopes before ranking.

Acceptance criteria:

- [ ] A group/channel turn cannot retrieve or write unrelated user memory.
- [ ] CLI turns keep a deterministic default scope.
- [ ] Gateway session `memory_scope` is connected to long-term memory storage.
- [ ] Existing debug prompt output shows scope for retrieved memories.
- [ ] Tests cover same-query retrieval under two different scopes.

Likely files:

- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/gateway/session.py`
- `src/alpha_agent/cli.py`

Checkpoint:

- [ ] Memory CRUD and retrieval tests pass.
- [ ] `alpha debug prompt` still works for sessions with and without gateway
  metadata.
- [ ] Documentation in `README.md` and `docs/TODO.md` matches the new scope
  behavior.

## Phase 2: Candidate Lifecycle And Review

Goal: make extracted memory auditable before it becomes durable long-term
context.

Tasks:

- [ ] Add `memory_candidates` with candidate type, proposed layer, content,
  weak structure, salience, confidence, scope, source message ids, status,
  reviewer metadata, and timestamps.
- [ ] Add `memory_decisions` or decision metadata for approve, reject, edit,
  auto-approve, promote, supersede, and delete actions.
- [ ] Change normal turn extraction to write candidates first.
- [ ] Add policy-controlled auto-approval only for explicit high-confidence
  cases, such as direct "remember" instructions in trusted scopes.
- [ ] Extend CLI review commands from one-shot preview toward stored candidate
  workflows: list, approve, reject, edit, and inspect source.
- [ ] Emit structured traces for candidate creation and decision outcomes.

Acceptance criteria:

- [ ] No semantic or episodic long-term memory is written without an explicit
  decision path.
- [ ] Rejected candidates remain auditable but never enter retrieval.
- [ ] Edited candidates preserve the original source message ids and decision
  history.
- [ ] Review commands can recover from a previous session instead of requiring
  the original message argument again.

Likely files:

- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/review.py`
- `src/alpha_agent/memory/persistence.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cli.py`
- `tests/test_memory_store.py`
- `tests/test_memory_review.py`

Checkpoint:

- [ ] Automatic extraction can be disabled, candidate-only, or auto-approve by
  config/policy.
- [ ] CLI review works without corrupting source transcripts.
- [ ] Prompt construction never sees pending or rejected candidates.

## Phase 3: Atomic Memory Lifecycle, Dedup, And Forgetting

Goal: let memory be corrected instead of silently accumulating stale or
contradictory facts.

Tasks:

- [ ] Replace the current semantic triple-only lifecycle with atomic memory
  records that include content, type, optional subject/predicate/object,
  entities, confidence, salience, stability, status, valid window, scope, and
  source ids.
- [ ] Preserve source ids on update or merge instead of replacing them.
- [ ] Add duplicate detection using exact weak-structure match, normalized
  content match, entity overlap, and retriever similarity.
- [ ] Add conflict detection for same subject/predicate under the same scope.
- [ ] Implement decision actions: store, skip, update, merge, supersede, and
  conflict-review.
- [ ] Add `forget this` and `forget memory id` support that marks memory deleted
  or superseded instead of physically removing it by default.
- [ ] Ensure retrieval filters inactive statuses.

Acceptance criteria:

- [ ] A corrected preference supersedes the old preference in retrieval.
- [ ] Audit output can show both the active memory and the old source evidence.
- [ ] Forgetting a memory removes it from prompt context immediately.
- [ ] Duplicate extraction does not create repeated prompt bullets.

Likely files:

- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/semantic.py`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/memory/review.py`
- `src/alpha_agent/cli.py`

Checkpoint:

- [ ] Status-aware retrieval tests pass.
- [ ] Conflict and supersession tests cover both same-object duplicates and
  changed-object corrections.
- [ ] README memory inspection docs include forget and audit behavior.

## Phase 4: Extraction And Consolidation Quality

Goal: improve what gets remembered without losing deterministic fallback
behavior.

Tasks:

- [ ] Introduce an extractor interface with deterministic and LLM-assisted
  implementations.
- [ ] Define strict JSON schema for extracted candidates: layer, memory type,
  content, entities, weak structure, confidence, stability, salience, source
  ids, sensitivity flags, and rationale.
- [ ] Use recent session context and retrieved active memories as extraction
  context, so the extractor can identify updates and contradictions.
- [ ] Add extraction policies for sensitive data, platform/system messages,
  group chat write restrictions, and explicit "do not remember" requests.
- [ ] Expand consolidation beyond episodic-to-semantic promotion: consolidate
  candidates, merge duplicates, promote stable repeated facts, and queue
  conflict review.
- [ ] Add configurable consolidation modes: manual, after N turns, and scheduled
  after a scheduler exists.

Acceptance criteria:

- [ ] LLM extraction can be enabled without changing the caller-facing
  `AlphaAgent.respond()` contract.
- [ ] Deterministic extraction remains available for tests and offline use.
- [ ] Group/system messages do not become semantic facts unless policy allows
  them.
- [ ] Consolidation creates fewer, higher-quality active memories instead of
  only increasing memory count.

Likely files:

- `src/alpha_agent/memory/extractor.py`
- `src/alpha_agent/memory/consolidation.py`
- `src/alpha_agent/memory/review.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/config.py`
- `tests/test_consolidation.py`
- new extractor tests under `tests/`

Checkpoint:

- [ ] Existing deterministic tests pass without network or provider credentials.
- [ ] LLM-assisted extractor has contract tests using a mock provider.
- [ ] Consolidation reports include promoted, merged, skipped, superseded, and
  conflict counts.

## Phase 5: Retrieval, Ranking, And Prompt Injection

Goal: improve recall and reduce prompt noise after memory scope and lifecycle
are safe.

Tasks:

- [ ] Split retrieval into candidate generation and ranking stages.
- [ ] Add score breakdowns for keyword, FTS, recency, salience, stability,
  access, scope priority, status, and source confidence.
- [ ] Add per-layer prompt budgets so semantic, episodic, procedural, and
  session context do not crowd each other out.
- [ ] Add source/status/confidence rendering to debug prompt and optional
  user-facing memory-use explanations.
- [ ] Add query expansion from active entities, current session task state, and
  high-confidence profile preferences.
- [ ] Add optional vector retrieval only after the non-vector retriever is
  scope/status aware and testable.

Acceptance criteria:

- [ ] Retrieval output explains why each memory was selected.
- [ ] Prompt context is bounded by configured budgets.
- [ ] Inactive, rejected, forgotten, and out-of-scope memories never reach the
  prompt.
- [ ] Vector retrieval can be disabled without changing memory semantics.

Likely files:

- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/runtime/prompt_builder.py`
- `src/alpha_agent/cli.py`
- `src/alpha_agent/config.py`
- `tests/test_retrieval.py`
- `tests/test_prompt_builder.py`

Checkpoint:

- [ ] Retrieval regression fixtures show improved expected-memory selection.
- [ ] Prompt snapshots remain readable and source-aware.
- [ ] README retrieval section documents the new ranking model.

## Phase 6: Session State, Scene, Persona, And Graph

Goal: add higher-level memory layers only after the atomic layer is reliable.

Tasks:

- [ ] Replace deterministic session compression text with a structured session
  state projection: current goal, decisions, open questions, pending tasks,
  user constraints, relevant files/entities, and last action.
- [ ] Add scene memory as topic/project summaries built from active atomic
  memories and source transcripts.
- [ ] Add persona/profile memory as a low-frequency, high-stability projection
  from reviewed semantic memories and scene summaries.
- [ ] Connect `entity_nodes` and `relation_edges` to extraction/consolidation
  only where they support retrieval, conflict detection, or audit.
- [ ] Add drill-down from persona/scene items back to atomic memories and source
  messages.

Acceptance criteria:

- [ ] Session summaries preserve task state better than message clipping.
- [ ] Persona updates are infrequent, source-backed, and reversible.
- [ ] Scene summaries reduce repeated retrieval noise for long-running topics.
- [ ] Graph data is justified by an actual query or audit use case.

Likely files:

- `src/alpha_agent/runtime/context_compression.py`
- `src/alpha_agent/runtime/session_context.py`
- `src/alpha_agent/memory/consolidation.py`
- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/store.py`
- `docs/memory_design.md`

Checkpoint:

- [ ] Long-session prompt tests show stable task continuity.
- [ ] Persona/scene retrieval never overrides explicit current user requests.
- [ ] Documentation explains the distinction between transcript, atomic memory,
  scene, and persona.

## Phase 7: Operations And Product UX

Goal: make memory behavior visible, correctable, and safe for daily use.

Tasks:

- [ ] Add "what do you remember about me?" and scoped memory inspection
  commands.
- [ ] Add channel responses that can optionally show memory confidence/source
  when a memory materially influenced the answer.
- [ ] Add memory audit commands for source messages, decisions, and supersession
  chains.
- [ ] Add policy configuration for channel-level memory capture defaults.
- [ ] Add maintenance commands for stale candidates, inactive memory cleanup,
  consolidation runs, and retrieval diagnostics.
- [ ] Add operational metrics for candidate volume, approval rate, conflict
  rate, retrieval hit rate, and forgotten memory count.

Acceptance criteria:

- [ ] A user can inspect, approve, correct, or forget memory from the CLI.
- [ ] Gateway adapters can expose the same review/forget flows later without
  duplicating core memory logic.
- [ ] Operators can diagnose why a response used a specific memory.
- [ ] Memory maintenance can run without changing active transcript history.

Likely files:

- `src/alpha_agent/cli.py`
- `src/alpha_agent/gateway/runner.py`
- `src/alpha_agent/memory/review.py`
- `src/alpha_agent/memory/store.py`
- `README.md`
- `docs/TODO.md`

## Non-Goals Until Earlier Phases Land

- Do not make vector retrieval the main project before scope, lifecycle, and
  candidate review exist.
- Do not build a full knowledge graph before atomic memory records are reliable.
- Do not auto-generate persona/profile memory from unreviewed candidates.
- Do not add compatibility shims for old memory data unless a concrete migration
  need exists.
- Do not add platform-specific memory behavior inside the core runtime; use
  normalized source metadata and policy instead.

## Documentation Updates Required By Implementation

Every phase that changes behavior should update:

- `README.md` for user-facing CLI/config behavior.
- `docs/TODO.md` for roadmap status.
- `docs/memory_design.md` only when the architecture intent changes, not for
  every implementation detail.
- Tests that define memory behavior, especially retrieval, review,
  consolidation, prompt building, and session context.

## Suggested Implementation Order

1. Phase 0: baseline tests and evaluation helpers.
2. Phase 1: scope model and controller boundary.
3. Phase 2: candidate lifecycle and review persistence.
4. Phase 3: atomic memory status, dedup, conflict, and forgetting.
5. Phase 4: extraction/consolidation quality.
6. Phase 5: retrieval/prompt improvements and optional vector module.
7. Phase 6: structured session state, scene, persona, and graph.
8. Phase 7: operations and product UX.
