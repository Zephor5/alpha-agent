# Legacy Removal Inventory

## Position

This plan removes, rather than adapts, every deterministic cognition subsystem that has no role in the target architecture. The target answer path uses only the runtime system prompt, the session profile snapshot, session history, and model-selected tools. The target cognition pipeline is entity-first beliefs plus LLM-mediated background integration. Any structure that exists only to serve the old deterministic-reflection / value-lens / strategy-guidance / context-window-summary model is deleted.

Reachability fact behind this decision, with one exception that must be handled in Phase 0: the strategy, procedure, context-window, value-lens, reflector, and `CognitionView` subsystems are reachable today only through CLI inspection commands and the manual `alpha cognition consolidate` worker set, which the daemon never runs automatically. Removing those does not touch the real answer path.

The one exception is the counterpart profile source. The answer path does have a live dependency on the digest: `AlphaAgent.respond` imports `active_counterpart_digest` (`agent.py` line 23) and `_session_profile_snapshot` (`agent.py` around line 743) reads it through `BeliefProjection(auto_rebuild=True)`. Deleting `counterpart_profile.py` (R7) therefore breaks the answer path unless the same Phase 0 change rewrites `_session_profile_snapshot` to read `summary_beliefs(summary_kind=counterpart_profile)` and return `None` when absent. R7 and that runtime migration are one atomic change; do not treat the digest removal as answer-path-neutral. Every other subsystem in this inventory is answer-path-neutral.

## Removal Rules

1. **Delete, do not deprecate.** No compatibility shim, no dormant module, no "kept for now" class. Existing database data is not preserved, so there is no migration to protect.
2. **Remove the whole vertical.** A subsystem removal includes its model, projection, schema table(s), event kinds, payload-contract validators, config keys, CLI commands, and the worker(s) that feed it.
3. **No test residue.** Delete the tests for removed behavior outright. Do not convert them into negative tests, "should-not-exist" assertions, "raises on legacy type" probes, or skipped/xfail markers. After removal there must be no test that names the removed symbol for any reason.
4. **Leave no dangling references.** `__init__.py` exports, `models/__init__.py` re-exports, `controller.default_projection_registry`, `payload_contract` dispatch tables, and docs must not mention removed symbols. A final `rg` for each removed name over `src tests` must be empty.
5. **Dead enum values go too.** Every `CognitiveEventKind` whose only producer/consumer is removed is deleted from `enums.py` in the same change.

## Keep List (do not remove)

These are load-bearing in the answer path or explicitly retained by this plan. Adapt them; do not delete them.

| Component | Why it stays |
| --- | --- |
| `cognitive_events` event log + `EventEmitter` | Demoted to non-canonical audit log, but still written for forensics. |
| `CounterpartProjection`, `counterpart_router`, `counterpart_view` | Counterpart identity drives session binding and profile targeting. |
| Belief storage (`BeliefProjection` → `atomic_beliefs` / `summary_beliefs`) | Becomes the primary entity store. Transformed by Phase 0.2, not deleted. |
| `session_messages`, `runtime_traces`, handover compression, `SessionContextAssembler`, `session_profile_snapshots` | Raw source material and answer-path continuity. |
| `archive_expired` worker | Deterministic expiration over explicit `valid_until` is explicitly allowed. Adapt to new lifecycle/validity fields. |
| Drive Loop + goals (`loops/drive.py`, `goals/`, `models/goal.py`, `GoalProjection`, `goal_view`, `GOAL_*` events, goal CLI) | Plan retains Drive Loop disabled by default; do not delete it. |
| `LoopCoordinator`, scheduler chunk/yield/checkpoint primitives | Reused for foreground/background coordination. Only the event-kind gating is replaced. |
| `render_counterpart_profile`, `wrap_system_reminder`, `estimate_chat_tokens`, `source_message_to_chat` | Text helpers used by the answer path. These are not the `CognitionView` renderers. |
| Manual `alpha cognition consolidate` entry point | Repointed at the new LLM pipeline; kept as operator/debug. |

## Removal Targets

Each target lists what to delete, why it has no target role, and what (if anything) replaces it.

### R1: Value subsystem

- Delete: `cognition/value/lens.py`, `cognition/value/resolver.py`, `cognition/value/profile_derivation.py`, `subject_value_lens` table, `ValueLens` / `ValueProfile` models, the `ValueKind` enum, the `value_profile` field on `Belief`, the `value_lens` field on `Subject`, `CognitiveEventKind.VALUE_LENS_SHIFTED`, the `cognition lens` CLI command group (`lens show` / `lens set` on the `lens_app` subapp), and the `value_lens_learning_threshold` / `value_lens_sensitivity_delta` fields on the `ConsolidationConfig` dataclass (they are dataclass fields, not `config.py` TOML keys).
- Why: the target ontology has no value-lens facet. `authority` ordering replaces value-weighted conflict resolution.
- Note: `derive_value_profile` is currently called inside `BeliefProjection` upsert and `workers/_common`. Removing the `value_profile` belief field is part of the Phase 0.1 model change, so this removal lands inside Phase 0, not later.

### R2: Strategy / legacy domain-guidance subsystem

- Delete: `models/strategy.py`, `projections/strategy.py`, `strategy_view`, `expire_strategies` worker, `STRATEGY_CHANGED` / `STRATEGY_EXPIRED` event kinds, and the `cognition strategies` / `cognition strategy-expire` CLI commands.
- Why: `strategy_view` is the old L2 domain-guidance record. Domain guidance becomes `summary_beliefs` with `summary_kind=domain_summary` (Phase 7).
- Replacement arrives in Phase 7. The capability is intentionally absent between removal and Phase 7.

### R3: Deterministic reflectors

- Delete: `reflectors/l2.py`, `reflectors/l2_rules/`, `reflectors/l3.py`, `reflectors/l3_aggregators/`, the self-model synthesis they drive (`SelfModel` deterministic population, `SELF_MODEL_UPDATED`, `BIAS_DETECTED`), and the `cognition reflect-l3` CLI command.
- Why: deterministic self-understanding and bias synthesis are exactly the "deterministic semantic synthesis" the plan forbids. Self-understanding becomes `summary_kind=self_memory_summary` (Phase 8); domain reflection becomes `domain_summary` (Phase 7).
- `ReflectorL2` is already never instantiated outside tests; `ReflectorL3` runs only via CLI and the manual worker set.
- The `LoopPriority` enum is kept (`REACTIVE`, `DRIVE`, `CONSOLIDATION` are still needed), but its `L2` and `L3` members become orphaned naming once the reflectors are gone. Remove `LoopPriority.L2` / `LoopPriority.L3`, and switch the loop-coordinator tests that use them as sample priorities (`test_loop_coordinator_yield.py`, `test_loop_coordinator_serial.py`) to surviving priority values rather than deleting those coordinator tests.

### R4: Procedure subsystem

- Delete: `models/procedure.py`, `projections/procedure.py`, `procedure_view`, `PROCEDURE_LEARNED` / `PROCEDURE_STRENGTHENED` / `PROCEDURE_WEAKENED` / `PROCEDURE_MATCHED` events, the `matched_procedure_ids` field carried through the context-window projection, and the procedure payload-contract validator.
- Why: a procedure is now `memory_kind=procedure` on an atomic belief. A separate procedure entity and `PROCEDURE_MATCHED` runtime hinting have no target role.

### R5: Cognition context-window subsystem

- Delete: `projections/context_window.py`, `context_window_view`, `context_window_background`, `models/context_window.py`, the `compress_context` worker, and `CONTEXT_COMPRESSED` / `CONTEXT_ANCHOR_SET` / `CONTEXT_ANCHOR_CLEARED` events.
- Why: this is cognition context compression, distinct from runtime handover compression. The answer path uses runtime handover compression from `session_messages`. Cognition context summaries must never enter prompts, and nothing else consumes them.
- This makes the Gap 8 / Phase 9 boundary trivial: after R5 there is only one compression mechanism left (runtime handover), so the only remaining work is naming and a guard test.

### R6: CognitionView renderers and their CLI surface

- Delete: `render/build_view.py`, `render/view.py` (`CognitionView`), `render/diff.py`, `render/evidence.py`, `render/graph_snapshot.py`, `render/base.py` (`RenderBudget` / `Renderer` / `RenderResult`), and the `cognition graph` / `cognition diff` / `cognition evidence` CLI commands. Update `render/__init__.py` to stop re-exporting the deleted renderer and `CognitionView` symbols.
- Keep: `render/text_chat.py`. Despite living under `render/`, it contains only answer-path chat helpers (`wrap_system_reminder`, `render_counterpart_profile`, `estimate_chat_tokens`, `source_message_to_chat`) imported by `runtime/agent.py` and `runtime/session_context.py`; it is not a `CognitionView` renderer. Optionally move it out of the `render/` package, but do not delete it.
- Why: `CognitionView` aggregates the projections removed in R2/R4/R5 and is banned from the answer path. `render/base.py` is used only by the deleted renderers and the deleted CLI commands, so it goes with them.
- Forensic inspection, where still wanted, reads current entity stores and audit logs directly rather than rebuilding a `CognitionView`.

### R7: Deterministic background workers replaced by LLM stages

- Delete: `merge_beliefs`, `summarize_counterpart`, `resolve_queued_conflicts`, `learn_value_lens` workers and remove them from `default_workers()`.
- Why: these are the deterministic versions of consolidation (Phase 4), counterpart profile summary (Phase 6), conflict review (Phase 4), and value learning (removed in R1). Keeping them would let the old rule-shaped path run alongside the target pipeline.
- `summarize_counterpart` also owns the `counterpart_digest:` belief object scheme and `counterpart_profile.py` digest helpers; remove those too. The profile snapshot source moves to `summary_kind=counterpart_profile` (see [Runtime Prompt And Memory Paths](03-runtime-memory-contract.md)).

### R8: Self-model and bias surface on the subject

- Delete: the deterministic `SelfModel` population path and the `self_model` synthesis in `subject` projection that R3 fed, the `SelfModel` and `ConfidenceCurve` model types, and the SelfModel inspection CLI (`cli.py` "Subject SelfModel" / "SelfModel History" tables).
- Why: self-understanding is `self_memory_summary` only. The subject record keeps identity/role; it does not keep a deterministically-synthesized self-model. `ConfidenceCurve` (a `"confidence=…;success=…"` string on capabilities) is part of this self-model surface and is unrelated to belief `confidence` in R9; it is removed here.

### R9: Numeric confidence scoring

- Delete (existing code): the `confidence: float` field on `Belief`, the `confidence` column on `belief_view` storage, the `_confidence_score` term in `memory_recall` scoring, and the `BELIEF_STRENGTHENED` / `BELIEF_WEAKENED` confidence-delta mechanics in `memory_propose` and the belief projection, plus any test asserting a confidence float or a strengthen/weaken delta.
- Constraint on new code: the Phase 2 background LLM output envelope must not introduce a `confidence` field. This is not an existing-code deletion; it is a guard so the removed score is not reintroduced through the new contract (already specified in [Background LLM Contract](02-background-llm-contract.md)).
- Why: there is no reliable way to obtain a calibrated confidence number. It is hardcoded on write (`0.72`), nudged by fixed deltas, or invented by the LLM. Source trust is expressed by `authority` (a categorical value reliably derived from the source channel); belief strength is expressed by evidence and validity. A made-up score adds false precision and a back door around the authority ceiling.
- Replacement: `reinforce` / strengthen becomes "record corroborating evidence and re-affirm the belief" (append sources, refresh `observed_at`), not "increment a float". Conflict resolution and recall ranking use authority, evidence count, and validity. `memory_recall` keeps its deterministic FTS match ranks (`term_rank` / `trigram_rank`), which are ephemeral search scores computed at query time, not stored belief strength.

### R10: L1 reflection subsystem

- Delete: `cognition/projections/reflection.py` (`ReflectionProjection`, `reflection_view`), `cognition/models/reflection.py`, `CognitiveEventKind.REFLECTED`, the `_validate_reflected` payload validator, and the `cognition reflections` CLI command.
- Why: no production code emits `REFLECTED`; the subsystem is already orphaned. Its only consumer was the L2 reflector (deleted in R3), and the answer path never references reflections. After R3 there is no producer and no semantic consumer.
- Note: this removes the deterministic L1 reflection projection only. The raw event log and runtime/tool traces remain the durable forensic record. If a reflection-style forensic view is wanted later, it is rebuilt from raw traces, not reintroduced as an `auto_rebuild` projection.

### R11: Belief lifecycle events as state-bearing records

- Delete: `BELIEF_FORMED`, `BELIEF_STRENGTHENED`, `BELIEF_WEAKENED`, `BELIEF_SUPERSEDED`, `BELIEF_RETRACTED`, `BELIEF_ARCHIVED`, `BELIEF_FORM_PENDING_CONFIRMATION`, `CONSOLIDATION_CONFLICT_QUEUED`, and `CONFLICT_KEPT_FOR_HUMAN_REVIEW` event kinds, the `BeliefProjection.apply` / `handles` event-consumption path, and their payload-contract validators.
- Why: once belief storage is direct entity state (Slice 0.2), these events have no state consumer. Belief lifecycle is the `lifecycle` field, written directly. A pending memory is a belief row with `lifecycle=pending_confirmation`, not a `BELIEF_FORM_PENDING_CONFIRMATION` event. A queued conflict is a ledger / conflict row (the processing ledger already defines a `conflict` source type), not a `CONSOLIDATION_CONFLICT_QUEUED` event.
- Adapt the kept `archive_expired` worker: it sets `lifecycle=archived` directly on the entity store instead of emitting `BELIEF_ARCHIVED`.
- Keep: `MEMORY_PROPOSED` as the single foreground audit event. Forensic audit of belief changes is a generic memory-audit record or the ledger stage-run records, not per-kind belief events.
- This makes concrete the foreground write-path migration already required in Slice 0.3: removing the event spine means removing these event kinds, not only stopping `memory_propose` from emitting them.

## Sequencing

R1, R4, R5, R6, R9, R10, R11 are deletions with no later replacement dependency and should land inside Phase 0 alongside the ontology refactor, because they share the `Belief` / `Subject` model edit and the `enums.py` / `payload_contract.py` / `controller.py` edits. R2, R3, R7, R8 remove capabilities whose LLM replacements arrive in Phases 4/6/7/8; they still land in Phase 0 (the deterministic versions are deleted up front), and the corresponding capability stays absent until its target phase. This is consistent with the plan's stance that existing data and partial capability are not preserved during the refactor. R11 in particular is the deletion side of the Slice 0.3 foreground write-path migration: the belief event kinds are removed as the entity store becomes primary.

## Completion Check

Legacy removal is complete when, for every removed symbol in R1-R11:

- `rg` over `src tests` returns no match. At minimum these patterns must all be empty:

  ```text
  ValueLens|ValueProfile|ValueKind|value_lens|value_profile
  StrategyProjection|strategy_view|STRATEGY_CHANGED|STRATEGY_EXPIRED
  ProcedureProjection|procedure_view|PROCEDURE_LEARNED|PROCEDURE_STRENGTHENED|PROCEDURE_WEAKENED|PROCEDURE_MATCHED
  ContextWindowProjection|context_window_view|context_window_background|CONTEXT_COMPRESSED|CONTEXT_ANCHOR_SET|CONTEXT_ANCHOR_CLEARED
  ReflectionProjection|reflection_view|CognitiveEventKind\.REFLECTED
  CognitionView|build_view|DiffRenderer|EvidenceRenderer|GraphSnapshotRenderer|RenderBudget
  ReflectorL2|ReflectorL3|SelfModel|ConfidenceCurve|SELF_MODEL_UPDATED|BIAS_DETECTED
  LoopPriority\.L2|LoopPriority\.L3
  counterpart_digest
  BELIEF_FORMED|BELIEF_STRENGTHENED|BELIEF_WEAKENED|BELIEF_SUPERSEDED|BELIEF_RETRACTED|BELIEF_ARCHIVED
  BELIEF_FORM_PENDING_CONFIRMATION|CONSOLIDATION_CONFLICT_QUEUED|CONFLICT_KEPT_FOR_HUMAN_REVIEW
  ```

- A separate, scoped check for numeric belief confidence (R9), because the bare word `confidence` legitimately survives elsewhere:

  ```text
  rg -n "\bconfidence\b" src/alpha_agent/cognition/models/belief.py \
    src/alpha_agent/cognition/projections/belief.py \
    src/alpha_agent/tools/memory_propose.py src/alpha_agent/tools/memory_recall.py
  ```

  This must be empty. Do not grep `confidence` across all of `src/alpha_agent/cognition`: the kept counterpart model carries `StyleHint.confidence` (`models/counterpart.py`), a communication-style field unrelated to belief strength, and provider code uses the word too.
- Kept-symbol exceptions so the grep block is not misread: the `LoopPriority` enum itself is kept (only `.L2` / `.L3` go); `MEMORY_PROPOSED` is kept as the foreground audit event; `render/text_chat.py` and its helpers (`render_counterpart_profile`, `wrap_system_reminder`, `estimate_chat_tokens`, `source_message_to_chat`) are kept; `StyleHint.confidence` on the counterpart model is kept.
- No test references the removed symbol, including negative or skipped tests.
- `controller.default_projection_registry`, `models/__init__.py`, `loops/workers/__init__.py`, `render/__init__.py`, and `payload_contract` dispatch contain no removed entries.
- `enums.py` contains no orphaned `CognitiveEventKind` value and no orphaned `LoopPriority` member.
- `schema.sql` contains no orphaned table for a removed projection.
