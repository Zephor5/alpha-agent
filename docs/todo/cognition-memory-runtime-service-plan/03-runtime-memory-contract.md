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

### Explicit Recall

The LLM calls `memory_recall` when it needs long-term memory beyond the visible session context.

Requirements:

- Recall searches active beliefs using the stable ontology.
- Recall can scope results to counterpart, global, self, project, or session memory as supported by the tool.
- Recall can filter by `memory_kind`.
- Summary memories should not duplicate profile snapshot context in ordinary recall unless explicitly requested.
- Recall results remain compact and source-linked.
