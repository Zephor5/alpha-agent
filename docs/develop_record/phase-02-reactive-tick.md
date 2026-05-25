# Phase 02 Reactive Tick Dev Note

## Status

Completed on 2026-05-25.

## Scope

Phase 02 connected the Reactive single tick to `AlphaAgent.respond()` and made
the turn runtime pass through the nine cognition stages:

```text
Perceive -> Attend -> Interpret -> Judge -> Decide -> Act -> Feedback -> Reflect -> Revise
```

Each successful tick emits the stage event chain with a shared `tick_id`.
Reactive acquisition is non-blocking: when another loop holds the
single-subject coordinator, `respond()` returns a busy result immediately,
does not preempt the holder, and does not write cognitive events or
conversation messages for the rejected stimulus.

## Stage Boundary Choices

- Perceive owns the conversion from external stimulus into Perception and
  carries `CounterpartRef` forward from the stimulus source.
- Attend and Interpret remain separate modules. Attend extracts the lightweight
  focus, while Interpret is the future attachment point for real belief recall
  and contradiction checks.
- Judge and Decide stay rule-light in this phase. Value conflict resolution and
  learned procedure selection remain future phases.
- Effector is the only stage that touches the outside world. After the P1
  review fix, the default Effector owns a bounded tool loop itself: one tool
  iteration followed by a final LLM round. `AlphaAgent.respond()` no longer
  pre-builds LLM input with `SessionContextManager` + `PromptBuilder`; it
  injects a runtime runner at the Reactive Effector boundary to persist the
  transcript and tool traces.
- Feedback records the local comparison between decision and outcome.
- Reflect and Revise are placeholders in Phase 02. They emit the stage events
  needed to preserve the full causal chain, but they do not create real
  reflections, revisions, or belief updates yet.

## Projection Choices

- `SubjectProjection` is replay/default backed and returns the single subject
  (`SUBJECT_SELF`) when no subject update exists.
- `BeliefProjection` remains a stub and returns no recalled beliefs. Real belief
  projection is Phase 03.
- `ProcedureProjection` remains a stub and returns no matched procedures until
  learned procedures exist in a later phase.
- `ContextWindowProjection` is a lightweight Phase 02 implementation: recent
  perceived events become foreground context, with no real background or
  recalled context. Full context projection is Phase 04.

## Deferred Work

- Renderer extraction is deferred to Phase 09. Until then, Effector builds the
  prompt from `decision/window` at the stage boundary and owns minimal message
  rendering for the bounded LLM/tool loop.
- `alpha debug prompt --trace` is implemented and can print the baseline prompt
  preview plus the recent cognitive event chain for the post-review Reactive
  design. It is not the final Phase 09 renderer output.
- Real reflection rules are deferred to Phase 05.
- Real belief revision and supersession are deferred until the belief system is
  available.
- Centralized stimulus routing for non-user-message inputs is deferred to the
  context/routing phase; Phase 02 handles user-message stimuli directly.
