# Implementation Phases

## Implementation Plan

### Phase 0: Refactor The Belief Ontology

Description:

Replace the overloaded `CognitiveType` model with the stable belief classification contract. This is the prerequisite for every other phase, but it is not a single-file rename. Current `Belief` records, `belief_view` schema, recall filters, memory proposal mapping, consolidation helpers, value/self-memory aggregators, and tests all assume `cognitive_type`.

Phase 0 is complete only when active code no longer uses `CognitiveType` or `cognitive_type` as the memory semantic contract. The slices below are execution ordering guidance inside one migration, not independent release gates. It is acceptable for intermediate tests to fail while this phase is in progress; verification is evaluated against the completed Phase 0 change.

Phase 0 also carries two scope items beyond the rename:

- **Storage inversion for beliefs.** Slice 0.2 makes `atomic_beliefs` and `summary_beliefs` primary tables, replacing `belief_view`-as-projection, and removes the `auto_rebuild` / event-replay path for beliefs. This is the entity-state inversion from [Current Implementation Gaps](04-current-gaps.md) Gap 5. Phase 2 then layers the shared write service, processing ledger, and authority ceiling on top of these tables; it does not redesign the storage. Keep that boundary explicit so belief storage is not built twice.
- **Legacy removal.** The deterministic value, strategy, procedure, context-window, reflector, and `CognitionView` subsystems are deleted in Phase 0, per [Legacy Removal Inventory](07-legacy-removal.md). They share the `Belief` / `Subject` model edit and the `enums.py` / `payload_contract.py` / `controller.py` edits, so removing them up front keeps later phases on a clean tree. Removal is clean: delete the tests too, with no negative or skipped test left referencing removed behavior.

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

Note: `_ids.py` currently defines `update_policy`, `applicability`, `structure`, and `derivation` as opaque `NewType(str)`. Enforcing the new invariants means promoting the fields that now carry validated structure (`scope`, `derivation_stage`, `authority`, `validity`, `relations`) into real enums or structured types. Adding members to `enums.py` is necessary but not sufficient.

Likely files:

- `src/alpha_agent/cognition/models/belief.py`
- `src/alpha_agent/cognition/models/enums.py`
- `src/alpha_agent/cognition/models/_ids.py`
- `tests/cognition/test_types_frozen.py`
- `tests/cognition/test_belief_projection_apply.py`

#### Slice 0.2: Replace Belief Storage And Indexes

Description:

Replace the persisted belief schema with direct cognition entity storage around the new ontology fields. Existing database compatibility is intentionally not preserved.

Acceptance criteria:

- `atomic_beliefs` and `summary_beliefs` are primary tables that are written directly, not projections rebuilt from `cognitive_events`. The `belief_view` projection, its `auto_rebuild`, and the belief event-replay path are removed.
- `atomic_beliefs` stores and indexes `memory_kind`, `derivation_stage`, `scope`, `authority`, and `lifecycle`.
- `summary_beliefs` stores and indexes `summary_kind`, `derivation_stage`, `scope`, `authority`, and `lifecycle`.
- `scope` and `memory_kind` are stored columns; nothing parses them out of the `object` string.
- Search and recall candidate collection no longer depends on `cognitive_type`.
- FTS still indexes `content`, `object`, `about`, and structured entity terms.
- `BeliefRecallParams` and `BeliefSearchParams` filter by `memory_kind`, `summary_kind`, `scope`, and lifecycle as needed.
- Audit logs are not required to rebuild current belief state.
- This slice creates the storage; the shared write service, ledger, and authority ceiling are Phase 2 and are not duplicated here.

Likely files:

- `src/alpha_agent/cognition/projections/belief.py` (becomes the direct entity store, or is replaced by a store module)
- `src/alpha_agent/state/schema.sql`
- `tests/cognition/test_belief_projection_apply.py` (rework or replace for direct-store writes)
- `tests/cognition/test_belief_projection_rebuild.py` (delete; there is no rebuild path)
- `tests/cognition/test_recall_entity_overlap.py`
- `tests/cognition/test_recall_about_explicit.py`

#### Slice 0.3: Update Foreground Memory Tools

Description:

Move `memory_propose` and `memory_recall` onto the new ontology without compatibility shims. Tool protocol types map directly to `memory_kind`.

Acceptance criteria:

- `memory_propose` writes atomic beliefs with `derivation_stage=tool_written`.
- `memory_propose` stamps a fixed authority by source channel: `user_asserted` for a plain user statement, and `system_defined` / `human_confirmed` only through their explicit flows. It does not accept a caller-proposed authority, so Phase 0 needs no overclaim-rejection logic. Full ceiling enforcement (rejecting a proposed authority above the source ceiling) arrives in Phase 2 with the shared state-write service, when the background LLM can propose authority.
- `constraint` writes remain `memory_kind=constraint`; they are not encoded as procedures or object prefixes.
- `memory_recall` accepts `memory_kind` filters and does not expose removed cognitive types. The `types` JSON-schema enum is updated to the new vocabulary; this is a `strict=True`, model-visible tool-contract change.
- Ordinary `memory_recall` excludes summary beliefs unless summary recall is explicitly requested, filtering by table / `summary_kind` rather than by `counterpart_digest:` / `counterpart_profile:` object prefixes.
- Candidate selection, duplicate detection, target validation, and output formatting use ontology fields instead of parsing `object` prefixes.
- The foreground write path is migrated off the belief event spine in this slice, not deferred to Phase 2. Because Slice 0.2 removed the belief projection-apply path, an emitted `BELIEF_FORMED` would no longer persist anything, so `memory_propose` must write beliefs directly to the Slice 0.2 entity store. Concretely:
  - `respond()` stops injecting `apply_cognitive_event` into the memory-tool context (`agent.py` around line 426), and `_apply_tool_cognitive_event` (`agent.py` around line 1400) is removed. The `belief_projection` injected into tools no longer uses `auto_rebuild`.
  - `memory_propose` stops emitting `BELIEF_FORMED` / `BELIEF_STRENGTHENED` / `BELIEF_SUPERSEDED` / `BELIEF_RETRACTED` as state-bearing events and instead calls the belief store directly. It may still write an audit-only `MEMORY_PROPOSED` record, which is forensic, not canonical state.
  - `memory_propose` writes no `confidence` (R9): the hardcoded `confidence=0.72` (`memory_propose.py` around line 518) is gone, and `reinforce` re-affirms and appends evidence instead of strengthening a score.
- Phase 2 later wraps this direct write in the shared state-write service, ledger, and authority ceiling; it does not re-add events as the belief spine. `memory_propose` is intentionally touched in both Slice 0.3 (ontology + direct write) and Phase 2 (shared service), but the event spine is gone after Slice 0.3.

Likely files:

- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/tools/memory_recall.py`
- `src/alpha_agent/runtime/agent.py` (remove `_apply_tool_cognitive_event`, the `apply_cognitive_event` injection, and `auto_rebuild` on the tool belief store)
- `tests/cognition/test_memory_propose_tool.py`
- `tests/cognition/test_memory_recall_tool.py`
- `tests/cognition/test_recall_by_counterpart.py`

#### Slice 0.4: Delete Legacy Deterministic Cognition Subsystems

Description:

Delete, do not adapt, the deterministic subsystems that have no role in the target direction. This is the Phase 0 execution of [Legacy Removal Inventory](07-legacy-removal.md) (R1-R11). Removing them here is what lets the remaining consumers drop `CognitiveType` entirely instead of porting dead rule code.

Acceptance criteria:

- The value subsystem (R1) is deleted: `cognition/value/`, `subject_value_lens`, `ValueLens` / `ValueProfile`, the `value_profile` belief field and `value_lens` subject field, `VALUE_LENS_SHIFTED`, value CLI, and `value_lens_*` config.
- The strategy subsystem (R2), deterministic reflectors (R3), procedure subsystem (R4), cognition context-window subsystem (R5), `CognitionView` renderers and their CLI commands (R6), the deterministic background workers `merge_beliefs` / `summarize_counterpart` / `resolve_queued_conflicts` / `learn_value_lens` (R7), and the deterministic self-model/bias surface (R8) are deleted.
- `counterpart_profile.py` digest helpers and the `counterpart_digest:` object scheme are removed; no remaining code derives a profile from `CognitiveType.CONCEPT`.
- The runtime profile-snapshot read is migrated in this same slice, not deferred to Phase 6. `respond()` imports `active_counterpart_digest` (`agent.py` line 23) and `_session_profile_snapshot` (`agent.py` around line 743) reads the digest through `BeliefProjection(auto_rebuild=True)`. After deletion, `_session_profile_snapshot` reads `summary_beliefs(summary_kind=counterpart_profile)` and returns `None` when none exists, and the `active_counterpart_digest` import is removed. The empty-snapshot window until Phase 6 generates the first profile summary is expected and acceptable.
- Numeric `confidence` (R9) is deleted: the belief `confidence` field/column, the `memory_recall` `_confidence_score` term, the background envelope `confidence`, and the `BELIEF_STRENGTHENED` / `BELIEF_WEAKENED` delta mechanics. `reinforce` is redefined to append evidence and re-affirm, not to nudge a float. Deterministic FTS search ranks in recall are kept.
- The L1 reflection subsystem (R10) is deleted: `ReflectionProjection`, `reflection_view`, `models/reflection.py`, `CognitiveEventKind.REFLECTED`, its payload validator, and the `cognition reflections` CLI command. Nothing emits `REFLECTED`, so there is no producer to preserve.
- The belief lifecycle events (R11) are removed as state-bearing records: `BELIEF_FORMED` / `BELIEF_STRENGTHENED` / `BELIEF_WEAKENED` / `BELIEF_SUPERSEDED` / `BELIEF_RETRACTED` / `BELIEF_ARCHIVED` / `BELIEF_FORM_PENDING_CONFIRMATION` / `CONSOLIDATION_CONFLICT_QUEUED` / `CONFLICT_KEPT_FOR_HUMAN_REVIEW`, plus `BeliefProjection.apply` / `handles` and their payload validators. Belief lifecycle is the `lifecycle` entity field; pending confirmation is `lifecycle=pending_confirmation`; a queued conflict is a ledger row. `archive_expired` sets `lifecycle=archived` directly. `MEMORY_PROPOSED` is kept as the foreground audit event.
- `default_workers()` retains only workers with a target role (for example `archive_expired`). Drive Loop and goals are not touched.
- `enums.py` has no orphaned `CognitiveEventKind` value; `payload_contract.py`, `controller.py`, `models/__init__.py`, `loops/workers/__init__.py`, and `render/__init__.py` have no removed references; `schema.sql` has no orphaned table for a removed projection.
- Active code has no `CognitiveType` import and no persisted `cognitive_type` column or payload dependency.
- Tests for removed subsystems are deleted, not converted into negative, "raises on legacy", skipped, or xfail tests. After this slice, `rg` over `src tests` for any removed symbol is empty.

Likely files:

- `src/alpha_agent/cognition/value/` (delete)
- `src/alpha_agent/cognition/reflectors/` (delete L2/L3 and aggregators)
- `src/alpha_agent/cognition/projections/strategy.py`, `procedure.py`, `context_window.py`, `reflection.py` (delete)
- `src/alpha_agent/cognition/projections/belief.py` (remove the `apply` / `handles` event-consumption path; belief writes are direct — coordinate with Slice 0.2)
- `src/alpha_agent/cognition/models/strategy.py`, `procedure.py`, `context_window.py`, `value.py`, `reflection.py` (delete)
- `src/alpha_agent/cognition/render/` (delete `build_view`, `view`, `diff`, `evidence`, `graph_snapshot`, `base`; keep `text_chat`, which holds the answer-path chat helpers used by `agent.py` / `session_context.py`)
- `src/alpha_agent/cognition/loops/workers/` (delete R7 workers; keep `archive_expired` but adapt it to set `lifecycle=archived` directly instead of emitting `BELIEF_ARCHIVED`)
- `src/alpha_agent/cognition/counterpart_profile.py` (delete)
- `src/alpha_agent/runtime/agent.py` (switch `_session_profile_snapshot` to read `summary_beliefs`; drop the `active_counterpart_digest` import)
- `src/alpha_agent/cognition/models/_ids.py` (remove `SelfModel`, `ConfidenceCurve`), `models/subject.py` and `projections/subject.py` (remove the `self_model` / `value_lens` surface)
- `src/alpha_agent/cognition/controller.py`, `payload_contract.py`, `models/enums.py` (remove orphaned `CognitiveEventKind` values and `LoopPriority.L2` / `L3`), `models/__init__.py`
- `src/alpha_agent/cli.py` (delete `graph`, `diff`, `evidence`, `reflect-l3`, `reflections`, `strategies`, `strategy-expire`, the `lens` value CLI, and the SelfModel tables)
- `tests/cognition/` (delete the tests for every removed subsystem; repoint `test_loop_coordinator_serial.py` / `test_loop_coordinator_yield.py` off `LoopPriority.L2` / `L3` rather than deleting them)

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
- `rg -n "CognitiveType|cognitive_type" src tests` returns nothing. Scope the grep to `src tests` only; the plan docs themselves discuss `CognitiveType`, so including `docs/` would always match and is not a valid gate. Update `docs/cognition` separately in this slice.
- The legacy-residue greps in the [Legacy Removal Inventory](07-legacy-removal.md) completion check all return nothing across `src tests`, confirming no removed symbol survives (including no negative or skipped tests). This covers value/strategy/procedure/context-window/reflector/`CognitionView` symbols, the orphaned `LoopPriority.L2` / `LoopPriority.L3` members, `SelfModel` / `ConfidenceCurve` types, and the removed `CognitiveEventKind` values.
- `rg -n "\bconfidence\b" src/alpha_agent/cognition/models/belief.py src/alpha_agent/cognition/projections/belief.py src/alpha_agent/tools/memory_propose.py src/alpha_agent/tools/memory_recall.py` returns nothing, confirming numeric belief `confidence` (R9) is gone. Do not widen this to all of `src/alpha_agent/cognition`: the kept `StyleHint.confidence` (counterpart communication style) is unrelated and would false-fail. The belief lifecycle events (R11) are covered by the residue grep block in the [Legacy Removal Inventory](07-legacy-removal.md) completion check.
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

Much of this codifies current behavior: `respond()` already excludes `CognitionView` and internal cognition state today. The new work is the guard tests plus collapsing the two prompt-construction paths into one.

Acceptance criteria:

- `cli prompt` and `respond()` build the answer prompt through one shared builder, so they cannot drift. The duplicated profile-context builder on each side is removed.
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

Define the current-state write boundary and the background LLM structured-output contract before implementing any cognitive worker. Belief storage was already made primary in Slice 0.2; this phase adds the shared write service, processing ledger, and authority-ceiling enforcement on top of that storage, with audit logs and indexes as support mechanisms. It does not redesign the tables.

Because the deterministic consolidation workers were deleted in Slice 0.4, `tests/cognition/test_consolidation_loop.py` no longer tests rule-shaped workers. Treat it as the background-pipeline test file that this and later phases grow, or rename it; do not restore the deterministic loop to keep an old test alive.

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
- Program logic owns project identity: a deterministic helper normalizes an LLM-derived project descriptor into a stable `project` reference id and mints it, and project-scoped beliefs carry that ref in `about`. The LLM never mints project ids (see [Belief Ontology](01-belief-ontology.md)). This helper lives in the state service so both foreground and background writes resolve the same project ref for the same descriptor.
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

This phase has a prerequisite in `runtime/context_handover.py`: the compact fast path needs the `handover_compression.completed` trace to carry the prompt prefix hash, tools schema hash, covered source refs / ordinal range, model, and extraction version. Today that trace records only `compression_point_ordinal`, `compression_version`, provider, message count, tool count, `tool_choice`, and the compressed-message id/ordinal. Extend the trace (and the assembler call site in `agent.py` that supplies tools) to emit the missing fields before, or as the first step of, this phase. Do this as a small leading slice rather than smuggling it into the worker.

Acceptance criteria:

- The `handover_compression.completed` trace records prompt prefix hash, tools schema hash, covered source message refs and ordinal range, model, and extraction version, enough for the worker to build a compact fast-path source window deterministically.
- Worker supports compact fast-path source windows created from `handover_compression.completed`.
- Compact fast-path extraction reconstructs the same stable messages/tools prefix as handover compression from durable inputs and changes only the suffix instruction.
- Worker supports inactive backlog source windows for sessions that have no active foreground turn, no pending handover maintenance for the same range, and no active compact extraction window for that range.
- Worker selects bounded source windows from raw session messages and runtime/tool traces.
- Worker records selected source windows in the processing ledger with exact source refs and a deterministic idempotency key.
- LLM extracts id-less atomic belief drafts using the stable ontology.
- LLM output is not required to provide precise source message ids; program logic attaches source-window provenance.
- Program logic generates belief ids only after validation.
- For a project-scoped draft, the LLM supplies only a project descriptor; program logic normalizes it to the stable `project` ref via the Phase 2 helper and sets that ref as the belief's `about` target. A project-scoped draft with no resolvable descriptor is rejected rather than written with an empty `about`, so `scope` and `about` stay paired.
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
- `uv run pytest tests/test_context_handover.py -q`

Likely files:

- `src/alpha_agent/runtime/context_handover.py` (record prefix hash, tools schema hash, source refs, model, extraction version on the completed trace)
- `src/alpha_agent/runtime/agent.py` (pass the tools schema / source-window info to the compression trace)
- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/` state service or store modules
- `tests/cognition/`
- `tests/test_context_handover.py`

### Phase 4: Build LLM Memory Consolidation And Conflict Review

Description:

Build LLM-mediated consolidation over extracted atomic beliefs and active beliefs. The LLM proposes semantic operations. Program logic validates program-attached provenance, target refs, authority ceilings, lifecycle transitions, and confirmation requirements.

Acceptance criteria:

- LLM proposes create, strengthen, supersede, retract, archive, or pending-confirmation operations. `strengthen` means re-affirm the belief and attach corroborating evidence; it does not increment a numeric score (there is none).
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

- `AlphaDaemon` creates exactly one `LoopCoordinator` for the subject and hands it to `AgentFactory`, which holds that single instance and injects it into every agent it creates. `AgentManager` caches one agent per session, so all cached agents and the background service must share the same coordinator object, not a per-agent default. `LoopCoordinator(SUBJECT_SELF)` is a single-subject lock; sharing one instance is the entire point. Today `AgentFactory.create()` passes no coordinator, so each agent falls back to its own default and cannot cooperate.
- `AgentFactory` injects the shared coordinator into every daemon-created `AlphaAgent`.
- Introduce target config `[cognition.background].enabled`, defaulting to `true`.
- Daemon creates one `BackgroundCognitionService` during startup.
- Daemon does not start automatic background ticks when `[cognition.background].enabled = false`.
- Daemon starts automatic background ticks by default when `[cognition.background].enabled = true`.
- Legacy `[cognition.consolidation]` settings are not the daemon automatic lifecycle switch. The new `[cognition.background]` section, its env passthrough, and the daemon status fields are added to the config surface: `config.py`, `config.example.toml`, and `.env.example`. The existing `test_config_set_preserves_cognition_consolidation_section` guard in `tests/test_config.py` is updated for the new section split (the deterministic-consolidation worker config it protected was gutted in Phase 0).
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
- `src/alpha_agent/daemon/status.py` (background lifecycle status fields)
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/consolidation.py`
- `src/alpha_agent/cognition/loops/scheduler.py`
- `src/alpha_agent/config.py`, `config.example.toml`, `.env.example` (add `[cognition.background]`)
- `README.md` and active cognition docs (document the daemon background switch)
- `tests/test_daemon_runtime.py`
- `tests/test_config.py` (update the consolidation-section guard for the new config split)

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

- `uv run pytest tests/cognition/test_domain_summary_worker.py -q` (new; the old `test_reflector_l2_phase08.py` is deleted with the L2 reflector)
- `uv run pytest tests/cognition/test_memory_propose_tool.py -q`
- `uv run pytest tests/test_agent_loop.py -q`

Likely files:

- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/cognition/` state service or summary belief store modules
- `src/alpha_agent/cognition/loops/workers/`
- `tests/cognition/test_memory_propose_tool.py`
- `tests/cognition/test_domain_summary_worker.py` (new)

### Phase 8: Generate Self-Memory Summaries

Description:

Build self-understanding synthesis as a separate background LLM stage. It consumes long-window evidence and memory summaries, then persists `summary_beliefs` with `summary_kind=self_memory_summary`. There is no separate cognition entity for this layer.

Acceptance criteria:

- Self-memory summary content is produced through LLM-mediated synthesis.
- Program logic validates the summary belief shape before writing `summary_beliefs`.
- Self-memory summaries do not enter runtime prompts by default.
- Tests cover malformed LLM output, program-attached provenance validation, and summary belief writes.

Verification:

- `uv run pytest tests/cognition/test_self_memory_summary_worker.py -q` (new; the old `test_l3_reflector.py` is deleted with the L3 reflector)
- `uv run pytest tests/cognition/test_memory_recall_tool.py -q`
- `uv run pytest tests/test_agent_loop.py -q`

Likely files:

- `src/alpha_agent/cognition/` state service or store modules
- `src/alpha_agent/cognition/loops/workers/`
- `tests/cognition/`

### Phase 9: Clarify Compression Boundaries

Description:

Confirm the distinction between runtime handover compression and background cognition integration in names, tests, and docs. The old cognition context-window compression (`compress_context`, `context_window_background`, `context_window_view`) was already deleted in Slice 0.4 (R5), so there is no second in-prompt compression mechanism left to confuse with handover compression. This phase is mostly a naming and guard-test confirmation, not a new build.

Acceptance criteria:

- Tests show runtime handover compression remains visible through `SessionContextAssembler`.
- Tests show background integration artifacts (extraction/consolidation/summary outputs) do not enter the answer prompt by default.
- Code comments or names make it clear that background summaries are cognition-maintenance artifacts, not answer-path context.

Verification:

- `uv run pytest tests/test_context_handover.py -q`
- `uv run pytest tests/test_session_context.py -q`
- `uv run pytest tests/cognition/test_consolidation_loop.py -q`

Likely files:

- `src/alpha_agent/runtime/context_handover.py`
- `src/alpha_agent/runtime/session_context.py`
- `src/alpha_agent/cognition/loops/workers/`
