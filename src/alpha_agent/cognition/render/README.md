# Cognition Renderers

Renderers convert a `CognitionView` into either provider payloads or deterministic
inspection text.

- `TextChatRenderer`: default turn-owned LLM prompt renderer for chat-completions
  messages.
- `GraphSnapshotRenderer`: Mermaid or DOT belief graph snapshot for inspection.
- `DiffRenderer`: event-kind delta between two turn ids for belief, value-lens,
  and strategy events currently present in the log.
- `EvidenceRenderer`: event chain for one belief id, including event inputs and
  outputs so audits can trace evidence back to perceptions when available.
