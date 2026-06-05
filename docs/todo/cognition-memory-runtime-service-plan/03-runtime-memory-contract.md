# Runtime Prompt And Memory Paths

## Accepted Runtime Prompt Contract

The real LLM request for an answer may include:

1. Runtime system message.
2. Profile snapshot message, if the session has one.
3. Session context, including runtime handover compression from `session_messages`.
4. Current user message.
5. Tool results from tools selected by the model.

The real LLM request must not include:

- Raw `CognitionView` dumps.
- `context_window_background` summaries.
- Domain-guidance summary beliefs by default.
- Self-memory summary beliefs by default.
- Hidden pre-turn memory recall results.
- Per-turn background summaries that were not requested through a tool.

`respond()` already excludes all of these today: it builds system message, profile-snapshot context, session history, and current user message, and `CognitionView` / `build_view` are reached only from CLI inspection commands. So this section largely codifies current behavior, and `CognitionView` itself is removed (see [Legacy Removal Inventory](07-legacy-removal.md), R6). The real risk is drift between two prompt-construction paths: `cli prompt` assembles the preview by hand, and `respond()` assembles through `PromptFrame` / `_rebuild_runtime_llm_messages`, with a duplicated profile-context builder on each side. Phase 1 must unify both on one shared prompt builder so they cannot diverge, rather than only asserting that two parallel implementations currently match.

## Memory Write Paths

### LLM-Initiated Memory Writes

The model uses `memory_propose` when the user explicitly asks to remember, correct, or forget something, or when the conversation makes a stable memory update clearly appropriate.

Requirements:

- `memory_propose` writes accepted updates into `atomic_beliefs`.
- `memory_propose` writes audit records for proposed, accepted, rejected, and pending-confirmation updates.
- The tool uses the stable belief ontology.
- The tool receives enough runtime context to link writes to session, turn, counterpart, and source message.
- Domain guidance summaries targeting `memory_propose` are enforced inside the memory proposal flow, not in the answer prompt.

### Background Memory Writes

Background cognition writes memory through LLM-mediated integration stages.

Requirements:

- Background workers use raw conversation records and runtime/tool traces as primary LLM inputs.
- LLM outputs are structured, validated, and persisted as cognition entities plus audit records.
- Background writes use the same belief ontology as `memory_propose`.
- Background writes respect authority and lifecycle invariants.
- Background writes can create `summary_beliefs` used by future session profile loading.

## Memory Read Paths

### Profile Snapshot

At session start or first counterpart binding, the runtime loads one stable profile-level memory snapshot for that session.

Requirements:

- The snapshot is built from `summary_beliefs` with `summary_kind=counterpart_profile`.
- The snapshot is stable within the session.
- Updated profile summaries affect new sessions or sessions without an existing snapshot.
- Existing session snapshots are not mutated mid-conversation.

This changes the snapshot source. Today the snapshot is created from the deterministic `counterpart_digest` belief (`active_counterpart_digest`, object prefix `counterpart_digest:`). That digest worker and its helpers are removed (see [Legacy Removal Inventory](07-legacy-removal.md), R7), and the LLM `counterpart_profile` summary that replaces it is generated in Phase 6. Between removal in Phase 0 and generation in Phase 6 there is no profile source, so new sessions get an empty snapshot. That gap is acceptable and intentional; it is not a regression to guard against.

The runtime read must switch in the same Phase 0 change that deletes the digest, not in Phase 6. `respond()` imports `active_counterpart_digest` and `_session_profile_snapshot` reads the digest through `BeliefProjection(auto_rebuild=True)`; once `counterpart_profile.py` is deleted that import breaks. So in Phase 0, `_session_profile_snapshot` is rewritten to read `summary_beliefs(summary_kind=counterpart_profile)` and return `None` when absent, the `active_counterpart_digest` import is removed, and the `memory_recall` summary exclusion stops matching `counterpart_digest:` / `counterpart_profile:` object prefixes and instead filters by table / `summary_kind`. The summary read simply returns nothing until Phase 6 starts producing profile summaries.

### Explicit Recall

The LLM calls `memory_recall` when it needs long-term memory beyond the visible session context.

Requirements:

- Recall searches active beliefs using the stable ontology.
- Recall can scope results to counterpart, global, self, project, or session memory as supported by the tool.
- Recall can filter by `memory_kind`.
- Summary memories should not duplicate profile snapshot context in ordinary recall unless explicitly requested.
- Recall results remain compact and source-linked.
