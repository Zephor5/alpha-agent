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
- [x] Add retrieval evaluation helpers that report retrieved ids, scores, score
  components, and whether expected memories were selected.
- [x] Add extraction evaluation helpers that compare candidates against expected
  type, content, scope, confidence, and source ids.
- [x] Add prompt inspection tests for memory source/status rendering.

Acceptance criteria:

- [x] `uv run pytest tests/test_memory_store.py tests/test_retrieval.py
  tests/test_prompt_builder.py tests/test_consolidation.py -q` passes.
- [x] A failing retrieval, extraction, or prompt-injection regression can be
  explained from test output without manually inspecting SQLite.
- [x] The evaluation fixtures do not depend on external LLM calls.

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

- [x] Add a `MemoryScope` domain model representing at least global user,
  platform user, chat/thread session, and project scopes.
- [x] Add scope columns or a normalized scope table for episodic, semantic,
  procedural, candidate, and access-log records.
- [x] Derive memory scope from CLI and gateway `source_metadata`.
- [x] Introduce a `MemoryController` or `MemoryPipeline` that owns capture,
  extraction, candidate persistence, promotion, retrieval policy, and trace
  emission.
- [x] Keep `AlphaAgent.respond()` as the readable orchestration path, but move
  memory policy decisions out of the runtime method body.
- [x] Update retrieval to filter by allowed scopes before ranking.

Acceptance criteria:

- [x] A group/channel turn cannot retrieve or write unrelated user memory.
- [x] CLI turns keep a deterministic default scope.
- [x] Gateway session `memory_scope` is connected to long-term memory storage.
- [x] Existing debug prompt output shows scope for retrieved memories.
- [x] Tests cover same-query retrieval under two different scopes.

Likely files:

- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/gateway/session.py`
- `src/alpha_agent/cli.py`

Checkpoint:

- [x] Memory CRUD and retrieval tests pass.
- [x] `alpha debug prompt` still works for sessions with and without gateway
  metadata.
- [x] Documentation in `README.md` and `docs/TODO.md` matches the new scope
  behavior.

## Phase 2: Candidate Lifecycle And Review

Goal: make extracted memory auditable before it becomes durable long-term
context.

Tasks:

- [x] Add `memory_candidates` with candidate type, proposed layer, content,
  weak structure, salience, confidence, scope, source message ids, status,
  reviewer metadata, and timestamps.
- [x] Add `memory_decisions` or decision metadata for Phase 2 actions: approve,
  reject, edit, auto-approve, and promote. Supersede/delete remain Phase 3.
- [x] Change normal turn extraction to write candidates first.
- [x] Add policy-controlled auto-approval only for explicit high-confidence
  cases, such as direct "remember" instructions in trusted scopes.
- [x] Extend CLI review commands from one-shot preview toward stored candidate
  workflows: list, approve, reject, edit, and inspect source.
- [x] Emit structured traces for candidate creation and decision outcomes.

Current status: `memory_decisions` exists for pending, approve, reject, edit,
auto-approve, and promote paths. Unpromotable approved candidates fail and roll
back instead of being treated as successful approvals. Supersede/delete decision
semantics are still Phase 3 follow-up work.

Acceptance criteria:

- [x] No semantic or episodic long-term memory is written without an explicit
  decision path.
- [x] Rejected candidates remain auditable but never enter retrieval.
- [x] Edited candidates preserve the original source message ids and decision
  history.
- [x] Review commands can recover from a previous session instead of requiring
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

- [x] Automatic extraction can be disabled, candidate-only, or auto-approve by
  config/policy.
- [x] CLI review works without corrupting source transcripts.
- [x] Prompt construction never sees pending or rejected candidates.

## Phase 3: Atomic Memory Lifecycle, Dedup, And Forgetting

Goal: let memory be corrected instead of silently accumulating stale or
contradictory facts.

Tasks:

- [x] Replace the current semantic triple-only lifecycle with atomic memory
  records that include content, type, optional subject/predicate/object,
  entities, confidence, salience, stability, status, valid window, scope, and
  source ids.
- [x] Preserve source ids on update or merge instead of replacing them.
- [x] Add duplicate detection using exact weak-structure match, normalized
  content match, entity overlap, and retriever similarity.
- [x] Add conflict detection for same subject/predicate under the same scope.
- [x] Implement decision actions: store, skip, update, merge, supersede, and
  conflict-review.
- [x] Add `forget this` and `forget memory id` support that marks memory deleted
  or superseded instead of physically removing it by default.
- [x] Ensure retrieval filters inactive statuses.

Acceptance criteria:

- [x] A corrected preference supersedes the old preference in retrieval.
- [x] Audit output can show both the active memory and the old source evidence.
- [x] Forgetting a memory removes it from prompt context immediately.
- [x] Duplicate extraction does not create repeated prompt bullets.

Likely files:

- `src/alpha_agent/memory/models.py`
- `src/alpha_agent/memory/schema.sql`
- `src/alpha_agent/memory/semantic.py`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/memory/review.py`
- `src/alpha_agent/cli.py`

Checkpoint:

- [x] Status-aware retrieval tests pass.
- [x] Conflict and supersession tests cover both same-object duplicates and
  changed-object corrections.
- [x] README memory inspection docs include forget and audit behavior.

## Phase 4: Extraction And Consolidation Quality

Goal: improve what gets remembered without losing deterministic fallback
behavior.

Tasks:

- [x] Introduce an extractor interface with deterministic and LLM-assisted
  implementations.
- [x] Define a local strict JSON validation schema for extracted candidates:
  layer, memory type, content, entities, weak structure, confidence, stability,
  salience, source ids, sensitivity flags, and rationale.
- [x] Use recent session context and retrieved active memories as extraction
  context, so the extractor can identify updates and contradictions.
- [x] Add extraction policies for sensitive data, platform/system messages,
  group chat write restrictions, and explicit "do not remember" requests.
- [x] Expand consolidation beyond episodic-to-semantic promotion: consolidate
  candidates, merge duplicates, promote stable repeated facts, and queue
  conflict review.
- [x] Add configurable consolidation modes: manual, after N turns, and scheduled
  after a scheduler exists.

Current status: deterministic extraction remains the default offline extractor.
`LLMAssistedMemoryExtractor` is provider-injected and contract-tested with a
mock provider; because the current provider interface has no structured-output
option, strictness is a local validation contract on the returned JSON object,
not provider-enforced structured output.
Runtime extraction now passes recent session messages and retrieved active
semantic memories into the extractor. Policy gates block explicit
do-not-remember requests, sensitive secrets, platform/system source messages,
and non-explicit group-chat writes. Consolidation now creates and processes
candidates through the controller candidate lifecycle and Phase 3 semantic
lifecycle, records decision audit rows, reports
promoted/merged/skipped/superseded/conflict counts, and keeps scheduled mode as
a no-op placeholder until a scheduler exists.

Acceptance criteria:

- [x] LLM extraction can be enabled without changing the caller-facing
  `AlphaAgent.respond()` contract.
- [x] Deterministic extraction remains available for tests and offline use.
- [x] Group/system messages do not become semantic facts unless policy allows
  them.
- [x] Consolidation creates fewer, higher-quality active memories instead of
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

- [x] Existing deterministic tests pass without network or provider credentials.
- [x] LLM-assisted extractor has contract tests using a mock provider.
- [x] Consolidation reports include promoted, merged, skipped, superseded, and
  conflict counts.

## Phase 5: Retrieval, Ranking, And Prompt Injection

Goal: improve recall and reduce prompt noise after memory scope and lifecycle
are safe.

Tasks:

- [x] Split retrieval into candidate generation and ranking stages.
- [x] Add score breakdowns for keyword, FTS, recency, salience, stability,
  access, scope priority, status, and source confidence.
- [x] Add per-layer prompt budgets so semantic, episodic, procedural, and
  session context do not crowd each other out.
- [x] Add source/status/confidence rendering to debug prompt and optional
  user-facing memory-use explanations.
- [x] Add query expansion from active entities, current session task state, and
  high-confidence profile preferences.
- [x] Add optional vector retrieval only after the non-vector retriever is
  scope/status aware and testable. No vector retriever was added in this phase;
  non-vector retrieval remains the only active path.

Acceptance criteria:

- [x] Retrieval output explains why each memory was selected.
- [x] Prompt context is bounded by configured budgets.
- [x] Inactive, rejected, forgotten, and out-of-scope memories never reach the
  prompt.
- [x] Vector retrieval can be disabled without changing memory semantics.

Likely files:

- `src/alpha_agent/memory/retrieval.py`
- `src/alpha_agent/memory/store.py`
- `src/alpha_agent/runtime/prompt_builder.py`
- `src/alpha_agent/cli.py`
- `src/alpha_agent/config.py`
- `tests/test_retrieval.py`
- `tests/test_prompt_builder.py`

Checkpoint:

- [x] Retrieval regression fixtures show improved expected-memory selection.
- [x] Prompt snapshots remain readable and source-aware.
- [x] README retrieval section documents the new ranking model.

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
