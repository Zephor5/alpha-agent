# Phase 04 ContextWindowProjection Completion Record

## Status

Completed.

## Delivered

- SQLite-backed `context_window_view` materializes foreground ContextWindow state.
- Foreground stores perception IDs and rebuilds from perceived event payloads.
- Anchors protect selected foreground perception IDs from ordinary rolling eviction.
- StimulusRouter centralizes stimulus-to-thread routing.
- Thread-local ContextWindow state is isolated by thread ID.

## Implementation Choices

- Raw perception content remains in the event log; the foreground window stores references.
- BeliefProjection recall is joined into the current tick's `ContextWindow.recalled` by the controller; the projection itself still owns only foreground window state.
- Background compression remains Phase 06.

## Follow-Up

- Keep Phase 05+ pending; this record only closes Phase 04.
