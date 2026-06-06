# Current Implementation Gaps

## Current Implementation Gaps To Close

### Gap 1: Belief Classification Is Not Stable

`CognitiveType` currently mixes content type, semantic facet, and abstraction level.

Impact:

- `constraint` is not first-class.
- `concept` is an overloaded catch-all.
- `causal`, `social`, and `temporal` are treated as peer types even though they should be relations, scope, or validity.
- Background integration cannot reliably classify or consolidate memory.

Target fix:

- Replace `CognitiveType` with the stable belief ontology.
- Update memory tools, cognition entity stores, indexes, and tests to use the new fields.
- Treat the belief ontology replacement as several ordered implementation slices, not one broad edit. The current model, persisted schema, memory tools, worker helpers, and tests are all coupled to `CognitiveType`.
- Do not preserve compatibility with old belief records.

### Gap 2: Background Cognition Is Too Rule-Shaped

Current consolidation workers are mostly deterministic and hand-authored around persistence operations.

Impact:

- The system performs cognition-like operations with code rules.
- Profile and domain-guidance generation risk becoming brittle heuristic outputs.
- The architecture cannot scale to richer cognition without multiplying rules.

Target fix:

- Convert background cognition workers into LLM-mediated integration workers.
- Keep deterministic code for orchestration, schema validation, lifecycle rules, indexing, and checkpoints.
- Generate cognitive content through LLM calls with strict structured outputs.

### Gap 3: Background Workers Are Manual

`ConsolidationLoop` and `Scheduler` exist, but the daemon does not run them automatically. The CLI can run consolidation manually, but a long-running runtime owner should also own automatic background cognition once the target LLM-mediated pipeline is ready.

The current configuration is also misleading: `[cognition.consolidation].enabled` defaults to `true`, but daemon startup does not use it to start a background scheduler. That makes the config look like an automatic lifecycle switch even though it is not one.

Impact:

- Background memory integration does not happen unless manually invoked.
- Profile summary memory may never be generated during normal use.
- Higher cognition layers remain stale.
- Users and implementers may assume daemon background work is running because a config flag says consolidation is enabled.
- The old deterministic consolidation loop could be accidentally wired into daemon auto-run before the LLM-mediated belief pipeline is ready.

Target fix:

- Add daemon-owned background cognition service.
- Introduce `[cognition.background].enabled`, defaulting to `true`, as the only daemon automatic background switch once the target LLM-mediated service exists.
- Reuse a shared `LoopCoordinator` across background workers and foreground `AlphaAgent` instances.
- Run background integration on configured intervals by default, and allow explicit disablement through `[cognition.background].enabled = false`.
- Expose background service state in daemon status.
- Keep manual cognition commands available as explicit operator/debug actions; they are not the daemon lifecycle contract.
- Do not auto-start legacy deterministic consolidation through daemon startup.
- Keep Drive Loop disabled by default unless explicitly configured.

### Gap 4: Runtime Agents Do Not Share Coordinator With Background Loops

`AlphaAgent` can receive a coordinator, but daemon-created agents currently use their own default coordinator. Background loops also create or receive their own coordinator.

Impact:

- Foreground and background loops cannot cooperate on priority.
- Background work cannot reliably yield to real user turns.

Target fix:

- Daemon creates one subject-level `LoopCoordinator`.
- `AgentFactory` injects it into every `AlphaAgent`.
- Background scheduler uses the same coordinator.

### Gap 5: Cognition State Writes Are Split, And The Store Is Event-Sourced

This is the largest structural gap. Belief state is not stored directly today; it is event-sourced. `cognitive_events` is canonical, and `belief_view` is a projection. `memory_propose` writes by emitting `BELIEF_FORMED` / `BELIEF_SUPERSEDED` / `BELIEF_STRENGTHENED` and applying them to the projection, and a fresh agent rebuilds `belief_view` from the event log through `BeliefProjection(..., auto_rebuild=True)`. Moving to entity-first storage is not a contained "add a state service" change; it inverts the persistence model.

Impact:

- Current cognition entity state can become coupled to the path that wrote it.
- Background and runtime code do not share one clear state-write contract.
- The current view/rebuild terminology makes audit logs look like canonical cognition state, which is not the target model.
- Inverting the model cuts the event spine that several consumers depend on. The cut bites in Phase 0, not Phase 2, because `memory_propose` stops emitting belief events as soon as belief storage becomes direct. The affected consumers are:
  - `Scheduler.should_run` / `ScheduleTrigger.watches`, which gate background work by counting belief events after a checkpoint.
  - The deterministic workers `summarize_counterpart`, `merge_beliefs`, and `resolve_queued_conflicts`, which write by emitting belief events and projecting them. These are removed (see [Legacy Removal Inventory](07-legacy-removal.md), R7), so the dependency disappears with them rather than being ported.
  - The CLI event trace and `evidence` renderer, which read the event log as if it were state. These are removed (R6) or repointed at audit-only reads.

Target fix:

- Introduce a `CognitionStateStore` or `MemoryStateService` used by memory tools, daemon, and background workers.
- The service persists current cognition entities directly, especially `atomic_beliefs` and `summary_beliefs`.
- The service maintains search indexes and lookup tables as implementation details.
- The service enforces the authority ceiling and lifecycle invariants in one place, for both foreground and background writes.
- The service writes audit logs for inspection but does not rely on audit logs to rebuild current cognition.
- Runtime reads persisted current entities directly and must not rebuild cognition during answer turns. Remove `auto_rebuild` and the projection-replay path for beliefs once storage is direct.
- Spell out the Phase 0.2 versus Phase 2 boundary so belief storage is not designed twice: Phase 0.2 makes `atomic_beliefs` / `summary_beliefs` the primary tables (replacing `belief_view`-as-projection); Phase 2 adds the shared write service, ledger, and ceiling enforcement on top of those tables.

### Gap 6: Profile Summary Generation Depends On Background Execution

Session profile snapshot loading already exists, but it depends on profile summary beliefs being present.

Impact:

- Profile snapshot loading is correct but often empty in normal runtime unless background integration has run.

Target fix:

- Make counterpart profile summary generation part of automatic LLM-mediated background integration.
- Add tests showing that after enough evidence and a background pass, a new session receives the generated profile snapshot.

### Gap 7: Domain Summary Consumption Is Incomplete

Some domain guidance exists in legacy guidance records today, but target tools and workers do not consistently consume the relevant guidance. In the target model, this guidance belongs in `summary_beliefs` with `summary_kind=domain_summary`.

Impact:

- Domain guidance summaries can exist without affecting their intended consumer.
- Adding them directly to the answer prompt would be the wrong fix.

Target fix:

- Generate domain guidance as `summary_beliefs` through LLM-mediated synthesis.
- Keep deterministic expiration and domain routing.
- Add explicit domain-summary consumption in target consumers, such as `memory_propose`.
- Do not render domain guidance summaries into prompts by default.

### Gap 8: Terminology Collision Around Compression

There are two different compression concepts:

- Runtime handover compression writes `compressed_message` into `session_messages` and is visible to the answer path.
- Cognition context compression writes background summaries for cognition context-window maintenance and should not be visible to the answer path.

Impact:

- It is easy to assume cognition background summaries should enter the prompt.

Target fix:

- Document and name these as separate mechanisms.
- Runtime handover compression remains answer-path continuity context.
- Cognition context compression is removed entirely, not renamed. The `compress_context` worker, `context_window_background`, and `context_window_view` have no answer-path consumer (see [Legacy Removal Inventory](07-legacy-removal.md), R5). After removal only one compression mechanism remains, so the Phase 9 work shrinks to naming clarity plus a guard test.

### Gap 9: Legacy Deterministic Cognition Subsystems Have No Target Role

Large parts of the current cognition tree exist only to serve the old deterministic model: value lenses, strategy guidance records, deterministic L2/L3 reflectors, procedure entities, cognition context-window summaries, and the `CognitionView` renderers. None of these are reachable from `respond()`, `SessionContextAssembler`, or `context_handover`; they run only through CLI inspection commands and the manual consolidation worker set, which the daemon never auto-runs.

Impact:

- They duplicate, in deterministic rule form, exactly the cognition the target pipeline produces with LLM-mediated summaries.
- Leaving them in place lets a rule-shaped path run beside the target pipeline and keeps `CognitiveType`, value profiles, and event-spine assumptions alive.

Target fix:

- Delete them outright with no compatibility shim and no test residue. The concrete keep/remove split, per-subsystem rationale, and completion check are in [Legacy Removal Inventory](07-legacy-removal.md).
- Land the deletions in Phase 0 so later phases start from a clean tree.
