# Checkpoints, Risks, And Done Definition

## Checkpoints

### Checkpoint A: Belief Ontology Accepted And Legacy Removed

Complete after Phase 0.

- The bottom-level belief contract is stable.
- Memory tools and cognition entity stores use the new classification.
- The old `CognitiveType` model no longer drives memory semantics.
- Belief storage is direct entity state; the belief projection-rebuild path is gone.
- The legacy deterministic subsystems (value, strategy, procedure, context-window, L2/L3 reflectors, L1 reflection projection, `CognitionView`), the state-bearing belief lifecycle events, and numeric `confidence` scoring are deleted with no source or test residue, per the Legacy Removal Inventory completion check.
- Source trust is carried by `authority` alone; no numeric belief-strength score remains.

### Checkpoint B: Runtime Contract Protected

Complete after Phase 1.

- The answer prompt shape is tested.
- Debug prompt and real prompt do not diverge.
- No arbitrary internal cognition state or audit log enters the answer prompt.

### Checkpoint C: State And Background LLM Boundaries Are Safe

Complete after Phase 2.

- Current cognition state has one shared write boundary.
- Audit logs are explicitly non-canonical.
- Background workers can call the LLM through the existing provider interface.
- Fixture-backed tests can run without live LLM calls.
- LLM output validation rejects malformed outputs, invented ids, invalid refs, authority overclaims, and prompt-injection content.

### Checkpoint D: Background Memory Integration Works

Complete after Phases 3 and 4.

- Background extraction creates validated id-less belief drafts and program-generated belief ids.
- Background extraction has both compact-adjacent fast path and inactive backlog fallback.
- Accepted extracted beliefs carry program-attached source-window provenance; LLM output is not required to provide exact source message ids.
- Background consolidation handles semantic operations and conflicts through LLM-mediated outputs.
- Deterministic code validates, persists current cognition entities, indexes, writes audit logs, and checkpoints.
- Processing ledger can report per-source, per-stage pending, processed, failed, and skipped records.

### Checkpoint E: Background Work Actually Runs

Complete after Phase 5.

- Daemon owns foreground/background coordination.
- Background integration is enabled by default through `[cognition.background].enabled = true`.
- `[cognition.background].enabled = false` explicitly disables automatic daemon background ticks.
- Foreground turns remain higher priority than background work.
- Background ticks run bounded eligible chunks, not mandatory full-pipeline passes.
- All LLM-mediated background stages are triggered by lower-layer quantity/change gates, not elapsed time alone.
- Daemon status reports background service lifecycle state.

### Checkpoint F: Profile Loop Is Closed

Complete after Phase 6.

- LLM can write memory.
- Background can integrate memory.
- Future sessions can load profile memory.
- LLM can explicitly recall consolidated memory.

### Checkpoint G: Domain Guidance Is Scoped

Complete after Phase 7.

- Domain guidance summary beliefs affect only target tools/workers.
- Domain guidance does not become a hidden prompt injection channel.

### Checkpoint H: Self-Memory Summary Is Isolated

Complete after Phase 8.

- Self-understanding is synthesized as `self_memory_summary` beliefs.
- Self-memory summaries do not enter answer prompts by default.

### Checkpoint I: Compression Boundary Is Clear

Complete after Phase 9.

- Runtime handover compression remains answer-path continuity context.
- Cognition background summaries remain background integration artifacts.
- Tests prevent background compression artifacts from entering prompts by default.

## Non-Goals

- Do not add a generic cognition prompt renderer to `respond()`.
- Do not auto-recall memory before every answer.
- Do not inject domain guidance summaries into prompts by default.
- Do not refresh session profile snapshots mid-session.
- Do not make background summaries answer-path context.
- Do not use deterministic rules for memory extraction, semantic classification, conflict interpretation, profile synthesis, self-memory synthesis, or domain-guidance synthesis.
- Do not preserve compatibility with existing database state while refactoring this subsystem.
- Do not keep the deterministic value, strategy, procedure, context-window, reflector, or `CognitionView` subsystems as dormant or "kept for now" code.
- Do not store a numeric `confidence` or belief-strength score, and do not let LLM output supply one. Source trust is `authority` only.
- Do not convert removed-behavior tests into negative, "raises on legacy", skipped, or xfail tests; delete them.
- Do not define production budget, cost, or rate-limit controls in this phase of the plan.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Belief ontology is under-specified | Later workers encode semantics inconsistently | Treat Phase 0 as the contract gate before background refactor |
| LLM background output is malformed | Memory corruption | Require structured output validation, source-window/provenance validation, and rejection paths |
| LLM background output invents ids | Broken references or duplicate records | Reject generated ids and let program logic generate ids |
| LLM background output overclaims authority | User or system rules can be weakened | Enforce authority ceilings from source type deterministically |
| Source processing progress is ambiguous | Messages may be skipped, duplicated, or impossible to debug | Track per-source and per-stage processing ledger entries beside raw records |
| LLM evidence refs are treated as precise source refs | Evidence becomes unreliable because rendered prompts may not expose durable ids | Attach source-window provenance in program logic; use LLM source notes only as optional orientation |
| Compact fast path duplicates inactive backlog work | The same raw messages produce duplicate belief drafts | Use one processing ledger and deterministic idempotency across both extraction paths |
| Prefix-cache optimization changes correctness | Extraction depends on provider cache behavior | Treat prefix cache as cost optimization only; cache miss must run the same validated pipeline |
| Background LLM cost or latency is uncontrolled | Resource use may grow after auto-run | Defer production budget controls until worker shape is stable; keep this visible as a follow-up |
| Background scheduler blocks foreground turns | User-visible latency | Use shared coordinator and short worker chunks; foreground remains reactive priority |
| Automatic background config implies behavior that does not exist | Users believe cognition is running when daemon never starts it | Make `[cognition.background].enabled = true` start the target daemon service and expose lifecycle state in daemon status |
| Legacy deterministic consolidation is wired into daemon auto-run too early | The target LLM-mediated memory architecture is bypassed | Keep manual consolidation separate; daemon auto-run starts only the target background service |
| Background refresh becomes time-driven | Stable memories are rewritten merely because time passed | Use lower-layer quantity/change gates for every LLM-mediated background stage |
| Profile memory becomes stale inside long sessions | Model may miss recent consolidated preferences | Keep session stability intentionally; rely on visible session history and explicit recall |
| Memory recall overreturns profile summary content | Duplicate or noisy answers | Exclude summary beliefs from ordinary recall unless explicitly requested |
| Domain guidance behavior is unclear | Domain guidance summaries either do nothing or leak into prompts | Require every domain guidance summary to declare and test a concrete target domain consumer |
| Audit log is mistaken for canonical cognition state | Engineers reintroduce log-sourced rebuilds | Treat audit logs as forensic records only; runtime reads current cognition entity stores |
| Current cognition state becomes stale after writes | Recall or profile loading misses newly persisted memory | Use one shared state-write service that persists entities and updates indexes in the same write path |
| Belief storage is designed twice across Slice 0.2 and Phase 2 | Wasted rework and conflicting schemas | Fix the boundary: 0.2 builds the primary tables and removes the rebuild path; Phase 2 only adds the write service, ledger, and ceiling on top |
| Legacy removal deletes load-bearing code | Answer path or daemon breaks | Follow the Keep List in the Legacy Removal Inventory; the answer path imports none of the removed subsystems, so verify each deletion against `respond()` / daemon usage before cutting |
| Removed behavior leaves test residue | Negative or skipped tests keep the dead contract alive and block a clean tree | Delete the tests with the code; the Phase 0 verification greps for removed symbols across `src tests` and must return nothing |
| Event-spine consumers break mid-Phase 0 | Scheduler gating or deterministic writers fail once belief events stop | Remove those consumers in the same slice (R7) and replace the event-kind gate with ledger gating; do not leave half-wired event watchers |

## Done Definition

The work is complete when normal daemon usage supports this loop:

1. The stable belief ontology is the single bottom-level memory contract.
2. User talks to Alpha.
3. LLM can write memory through `memory_propose`.
4. Daemon background integration runs by default through `[cognition.background].enabled = true`.
5. Daemon background integration uses LLM calls to compile raw messages and traces.
6. Background integration produces consolidated beliefs and profile summary beliefs when evidence is sufficient.
7. A later session loads profile memory at session start.
8. During any turn, the LLM can explicitly call `memory_recall` for additional long-term memory.
9. Domain guidance summaries affect only their target domain consumers.
10. Self-understanding is synthesized as self-memory summary beliefs and does not enter prompts by default.
11. No other cognition state enters the answer prompt by default.
12. The legacy deterministic cognition subsystems are gone, with no source or test residue.
