# Implementation Phases

## Implementation Plan

### Phase 0: Refactor The Belief Ontology

Description:

Replace the overloaded `CognitiveType` model with the stable belief classification contract. This is the prerequisite for every other phase, but it is not a single-file rename. Current `Belief` records, `belief_view` schema, recall filters, memory proposal mapping, consolidation helpers, value/self-memory aggregators, and tests all assume `cognitive_type`.

Phase 0 is complete only when active code no longer uses `CognitiveType` or `cognitive_type` as the memory semantic contract. The slices below are execution ordering guidance inside one migration, not independent release gates. It is acceptable for intermediate tests to fail while this phase is in progress; verification is evaluated against the completed Phase 0 change.

#### Slice 0.1: Define The New Belief Contract

Description:

Introduce the new ontology fields and enforce shape invariants at the model boundary before touching persistence or tools.

Acceptance criteria:

- The `Belief` family distinguishes `AtomicBelief` and `SummaryBelief` records.
- `AtomicBelief` records distinguish `memory_kind`, `derivation_stage`, `scope`, `authority`, `validity`, and `lifecycle`.
- `SummaryBelief` records distinguish `summary_kind`, `derivation_stage`, `scope`, `authority`, `validity`, and `lifecycle`.
- `constraint` is a first-class `memory_kind`.
- `concept`, `temporal`, `causal`, `social`, and `meta` are removed as top-level belief content types.
- Atomic beliefs require `memory_kind` and reject `summary_kind`.
- Summary beliefs require `summary_kind` and reject `memory_kind`.
- `scope` requires matching typed `about` references where applicable.
- `derivation_stage` is provenance only; it is not used as content classification.

Likely files:

- `src/alpha_agent/cognition/models/belief.py`
- `src/alpha_agent/cognition/models/enums.py`
- `src/alpha_agent/cognition/models/_ids.py`
- `tests/cognition/test_types_frozen.py`
- `tests/cognition/test_belief_state_apply.py`

#### Slice 0.2: Replace Belief Storage And Indexes

Description:

Replace the persisted belief schema with direct cognition entity storage around the new ontology fields. Existing database compatibility is intentionally not preserved.

Acceptance criteria:

- `atomic_beliefs` stores and indexes `memory_kind`, `derivation_stage`, `scope`, `authority`, and `lifecycle`.
- `summary_beliefs` stores and indexes `summary_kind`, `derivation_stage`, `scope`, `authority`, and `lifecycle`.
- Search and recall candidate collection no longer depends on `cognitive_type`.
- FTS still indexes `content`, `object`, `about`, and structured entity terms.
- `BeliefRecallParams` and `BeliefSearchParams` filter by `memory_kind`, `summary_kind`, `scope`, and lifecycle as needed.
- Audit logs are not required to rebuild current belief state.

Likely files:

- `src/alpha_agent/cognition/`
- `src/alpha_agent/state/schema.sql`
- `tests/cognition/test_belief_state_apply.py`
- `tests/cognition/test_belief_state_store.py`
- `tests/cognition/test_recall_entity_overlap.py`
- `tests/cognition/test_recall_about_explicit.py`

#### Slice 0.3: Update Foreground Memory Tools

Description:

Move `memory_propose` and `memory_recall` onto the new ontology without compatibility shims. Tool protocol types map directly to `memory_kind`.

Acceptance criteria:

- `memory_propose` writes atomic beliefs with `derivation_stage=tool_written`.
- `memory_propose` records accepted direct user-statement memories with an authority no higher than `user_asserted`.
- `constraint` writes remain `memory_kind=constraint`; they are not encoded as procedures or object prefixes.
- `memory_recall` accepts `memory_kind` filters and does not expose removed cognitive types.
- Ordinary `memory_recall` excludes summary beliefs unless summary recall is explicitly requested.
- Candidate selection, duplicate detection, target validation, and output formatting use ontology fields instead of parsing `object` prefixes.

Likely files:

- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/tools/memory_recall.py`
- `src/alpha_agent/runtime/agent.py`
- `tests/cognition/test_memory_propose_tool.py`
- `tests/cognition/test_memory_recall_tool.py`
- `tests/cognition/test_recall_by_counterpart.py`

#### Slice 0.4: Update Current Cognition Consumers

Description:

Remove the old enum from all active consumers so later background work starts from one contract. This includes deterministic workers that will be replaced by LLM-mediated behavior in later phases; for Phase 0 they must either use the new fields or be explicitly removed from default execution.

Acceptance criteria:

- Consolidation helpers and workers no longer group, summarize, or derive value profiles from `CognitiveType`.
- Existing counterpart digest/profile code no longer uses `CognitiveType.CONCEPT`; any remaining profile summary record is a `summary_belief` with `summary_kind=counterpart_profile`.
- Value/self-memory aggregators refer to `memory_kind` or `summary_kind` where they still consume beliefs.
- Active code has no `CognitiveType` import and no persisted `cognitive_type` column or payload dependency.
- Any deterministic semantic synthesis retained only as a temporary worker is named and documented as a later-phase removal target.

Likely files:

- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/counterpart_profile.py`
- `src/alpha_agent/cognition/value/profile_derivation.py`
- `src/alpha_agent/cognition/reflectors/l3_aggregators/`
- `tests/cognition/`

#### Slice 0.5: Update Tests, Fixtures, And Active Documentation

Description:

Finish the refactor by replacing old test helpers and current-state docs. Historical archive docs remain untouched unless explicitly required.

Acceptance criteria:

- Shared test helpers construct beliefs with the new ontology fields.
- Tests assert the new invariant failures and new recall filters directly.
- Active docs no longer describe `CognitiveType` as the current memory contract.
- `docs/develop_record/` is not used as implementation evidence for this refactor.
- Full project validation passes after the ontology replacement.

Phase 0 verification:

- `uv run pytest tests/cognition/test_types_frozen.py -q`
- `uv run pytest tests/cognition/test_memory_propose_tool.py -q`
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`
- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `rg -n "CognitiveType|cognitive_type" src tests docs/cognition docs/todo --glob '!docs/develop_record/**'`
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`

Likely files:

- `tests/cognition/`
- `docs/cognition/cognition_from_scratch.md`
- `docs/cognition/memory_design.md`
- `docs/cognition/cognition.md`
- `README.md`

### Phase 1: Lock The Runtime Contract

Description:

Codify that the real answer prompt contains system, profile snapshot, session history, current turn, and model-selected tool results. Add tests that prevent accidental internal cognition state injection.

Acceptance criteria:

- Tests assert that `respond()` does not include context-window background, domain guidance summaries, self-memory summaries, audit logs, or internal entity dumps by default.
- Tests assert that profile snapshot remains visible before session history.
- `debug prompt` output matches the same prompt contract as real runtime prompt construction.

Verification:

- `uv run pytest tests/test_agent_loop.py -q`
- `uv run pytest tests/test_cli_agent_loop.py -q`

Likely files:

- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cli.py`
- `tests/test_agent_loop.py`
- `tests/test_cli_agent_loop.py`

### Phase 2: Define Cognition State Store And Background LLM Contract

Description:

Define the current-state write boundary and the background LLM structured-output contract before implementing any cognitive worker. This phase makes `atomic_beliefs` and `summary_beliefs` the primary cognition state, with audit logs, indexes, and compiled domain controls as support mechanisms.

Acceptance criteria:

- A shared `CognitionStateStore` or `MemoryStateService` owns direct writes to current cognition entities.
- The state service updates search indexes and lookup tables as part of the same write path.
- The state service can write audit records for debugging, but audit records are not used as canonical rebuild input.
- Runtime memory tools and background workers use the same state-write boundary.
- Background workers use the same LLM interface and credentials as the foreground runtime.
- Tests can run background LLM workers with deterministic mock/fixture outputs.
- A sidecar processing ledger records per-source and per-stage status without mutating raw `session_messages` or traces.
- The ledger can answer which raw messages/traces are pending, claimed, processed, failed, or skipped for each stage.
- Stage checkpoints advance only after ledger updates and accepted cognition writes commit atomically.
- Create operations use id-less `atomic_belief_draft` or `summary_belief_draft` records as appropriate for the worker stage.
- Update operations use explicit target refs and can only reference cognition entity ids included in the LLM input.
- LLM output cannot provide new ids, idempotency keys, or authoritative source message refs.
- Program logic attaches source-window provenance to accepted outputs.
- Authority ceilings are enforced from source type.
- Malformed output, invented refs, invented ids, prompt-injection content, and authority overclaims are rejected without writing cognition entities or advancing checkpoints.
- LLM timeout or retry exhaustion does not write cognition entities or advance checkpoints.

Verification:

- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `uv run pytest tests/cognition/ -q`

Likely files:

- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/loops/consolidation.py`
- `src/alpha_agent/cognition/models/`
- `src/alpha_agent/cognition/` state service or store modules
- `src/alpha_agent/llm/`
- `tests/cognition/`

### Phase 3: Build LLM Memory Extraction Worker

Description:

Build the first LLM-mediated background worker. It selects raw source windows and asks the LLM to extract id-less atomic belief drafts. It does not consolidate, merge, or summarize. The preferred active-session path is compact-adjacent extraction that reuses the provider-visible prefix from runtime handover compression; inactive backlog extraction remains the fallback for sessions that do not compact.

Acceptance criteria:

- Worker supports compact fast-path source windows created from `handover_compression.completed`.
- Compact fast-path extraction reuses the same stable messages/tools prefix as handover compression and changes only the suffix instruction.
- Worker supports inactive backlog source windows for sessions that have no active foreground turn, no pending handover maintenance for the same range, and no active compact extraction window for that range.
- Worker selects bounded source windows from raw session messages and runtime/tool traces.
- Worker records selected source windows in the processing ledger with exact source refs and a deterministic idempotency key.
- LLM extracts id-less atomic belief drafts using the stable ontology.
- LLM output is not required to provide precise source message ids; program logic attaches source-window provenance.
- Program logic generates belief ids only after validation.
- Extracted beliefs use `derivation_stage=background_extracted`.
- Worker rejects outputs outside the selected source window.
- Worker marks source records processed for extraction only after validated outputs and ledger updates commit.
- Failed or malformed output leaves source records retryable and records failure details in the ledger.
- `compressed_message` may appear in extraction context for prefix stability but is never attached as cognition evidence.
- Existing beliefs or previous extraction coverage may be appended after the shared prefix to prevent duplicate drafts; actual update/supersede decisions remain consolidation work.
- Worker is fixture-testable in CI.

Verification:

- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`

Likely files:

- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/` state service or store modules
- `tests/cognition/`

### Phase 4: Build LLM Memory Consolidation And Conflict Review

Description:

Build LLM-mediated consolidation over extracted atomic beliefs and active beliefs. The LLM proposes semantic operations. Program logic validates program-attached provenance, target refs, authority ceilings, lifecycle transitions, and confirmation requirements.

Acceptance criteria:

- LLM proposes create, strengthen, supersede, retract, archive, or pending-confirmation operations.
- Update-like operations can only target beliefs included in the LLM input.
- Program logic rejects invalid lifecycle transitions.
- Program logic rejects authority overclaims instead of silently downgrading.
- Conflict review can mark an operation as `requires_confirmation`.
- Accepted consolidated beliefs use `derivation_stage=background_consolidated`.
- Failed or rejected LLM outputs do not corrupt memory state.

Verification:

- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`

Likely files:

- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/` state service or store modules
- `src/alpha_agent/tools/memory_recall.py`
- `tests/cognition/`

### Phase 5: Wire Daemon Coordination And Automatic Background Runner

Description:

Make daemon the owner of foreground/background loop coordination and automatic background integration. This is one execution boundary: background cognition only matters in normal runtime once it both shares foreground priority control and actually ticks without manual CLI invocation.

Acceptance criteria:

- `AlphaDaemon` creates or receives one shared `LoopCoordinator`.
- `AgentFactory` injects the shared coordinator into every daemon-created `AlphaAgent`.
- Introduce target config `[cognition.background].enabled`, defaulting to `true`.
- Daemon creates one `BackgroundCognitionService` during startup.
- Daemon does not start automatic background ticks when `[cognition.background].enabled = false`.
- Daemon starts automatic background ticks by default when `[cognition.background].enabled = true`.
- Legacy `[cognition.consolidation]` settings are not the daemon automatic lifecycle switch.
- The default-on background service must be the LLM-mediated target service, not the legacy deterministic consolidation loop.
- Background service supports `startup_delay_seconds`, `interval_seconds`, and bounded tick timeout behavior.
- Background service treats `interval_seconds` as a check cadence, not as a semantic refresh trigger.
- Each tick runs eligible bounded chunks and is not required to complete every pipeline stage.
- Every LLM-mediated background stage is gated by new/changed lower-layer material, starting from raw messages and traces.
- Summary generation uses initial, changed-source, and invalidated-source quantity gates.
- No LLM-mediated background stage has a time-only refresh gate.
- Background service stops cleanly on daemon shutdown.
- Graceful shutdown stops future ticks and lets the current chunk finish or yield.
- Immediate shutdown stops future ticks and requests the current chunk to yield as soon as possible.
- Background integration uses persistent `CheckpointStore`.
- Background integration uses the shared coordinator and yields to foreground turns.
- Background integration writes cognition entities through the shared state service, not through log replay.
- Daemon status exposes background enabled/running state, last tick, last success, last error, and next tick time.
- Tests prove a background holder can make a foreground turn return busy, and foreground priority can preempt or defer background chunks where applicable.
- Manual `alpha cognition consolidate --now` remains available and unchanged in purpose.
- Drive Loop remains disabled by default and is not started unless a separate explicit config enables it.

Verification:

- `uv run pytest tests/test_daemon_runtime.py -q`
- `uv run pytest tests/test_cli_daemon.py -q`
- `uv run pytest tests/test_agent_loop.py -q`
- `uv run pytest tests/cognition/test_consolidation_loop.py -q`

Likely files:

- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/daemon/manager.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/consolidation.py`
- `src/alpha_agent/cognition/loops/scheduler.py`
- `tests/test_daemon_runtime.py`

### Phase 6: Generate Profile Summaries And Load Session Snapshots

Description:

Ensure background integration can generate counterpart profile summary beliefs during normal runtime, and ensure future sessions load them through the existing session profile snapshot mechanism.

Acceptance criteria:

- Given sufficient counterpart-scoped evidence, background integration persists a `counterpart_profile` summary belief.
- Profile summary beliefs are persisted in `summary_beliefs` with `summary_kind=counterpart_profile` and `derivation_stage=background_summarized`.
- A new session bound to that counterpart creates a session profile snapshot from that summary.
- Existing sessions keep their original snapshot.
- Profile summary beliefs are not duplicated into ordinary `memory_recall` results.

Verification:

- `uv run pytest tests/test_agent_loop.py -q`
- `uv run pytest tests/cognition/test_consolidation_loop.py -q`
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`

Likely files:

- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/tools/memory_recall.py`
- `tests/test_agent_loop.py`
- `tests/cognition/test_consolidation_loop.py`

### Phase 7: Generate Domain Summaries And Enforce Domain Consumers

Description:

Generate domain guidance as `summary_beliefs` and wire target consumers to read only the summaries that apply to their domain. Do not make domain guidance a general prompt input.

Acceptance criteria:

- Domain guidance content is produced through LLM-mediated summary belief synthesis, not deterministic semantic rules.
- Expiration and domain routing remain deterministic.
- `domain_summary` beliefs targeting `memory_propose` affect memory proposal behavior.
- `domain_summary` beliefs targeting background learning affect only those background workers.
- Domain summaries with target domains unrelated to answer generation do not appear in runtime prompts.
- Tests cover active, expired, and unrelated domain summaries.

Verification:

- `uv run pytest tests/cognition/test_reflector_l2_phase08.py -q`
- `uv run pytest tests/cognition/test_memory_propose_tool.py -q`
- `uv run pytest tests/test_agent_loop.py -q`

Likely files:

- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/cognition/` state service or summary belief store modules
- `src/alpha_agent/cognition/loops/workers/`
- `tests/cognition/test_memory_propose_tool.py`
- `tests/cognition/test_reflector_l2_phase08.py`

### Phase 8: Generate Self-Memory Summaries

Description:

Build self-understanding synthesis as a separate background LLM stage. It consumes long-window evidence and memory summaries, then persists `summary_beliefs` with `summary_kind=self_memory_summary`. There is no separate cognition entity for this layer.

Acceptance criteria:

- Self-memory summary content is produced through LLM-mediated synthesis.
- Program logic validates the summary belief shape before writing `summary_beliefs`.
- Self-memory summaries do not enter runtime prompts by default.
- Tests cover malformed LLM output, program-attached provenance validation, and summary belief writes.

Verification:

- `uv run pytest tests/cognition/test_l3_reflector.py -q`
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`
- `uv run pytest tests/test_agent_loop.py -q`

Likely files:

- `src/alpha_agent/cognition/` state service or store modules
- `src/alpha_agent/cognition/loops/workers/`
- `tests/cognition/`

### Phase 9: Clarify Compression Boundaries

Description:

Make the distinction between runtime handover compression and cognition background integration explicit in names, tests, and docs.

Acceptance criteria:

- Tests show runtime handover compression remains visible through `SessionContextAssembler`.
- Tests show background integration artifacts do not enter answer prompt by default.
- Code comments or names make it clear that background summaries are cognition-maintenance artifacts, not answer-path context.

Verification:

- `uv run pytest tests/test_context_handover.py -q`
- `uv run pytest tests/test_session_context.py -q`
- `uv run pytest tests/cognition/test_consolidation_loop.py -q`

Likely files:

- `src/alpha_agent/runtime/context_handover.py`
- `src/alpha_agent/runtime/session_context.py`
- `src/alpha_agent/cognition/loops/workers/`
