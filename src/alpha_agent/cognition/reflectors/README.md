# Cognition Reflectors

Phase 05 implements L1 as a deterministic, read-only audit over one completed
Reactive tick. Phase 08 adds deterministic L2 strategy control. Phase 11 adds
deterministic L3 SelfModel aggregation.

| Level | Reads | Writes | Cadence |
| --- | --- | --- | --- |
| L1 | One completed Reactive tick | `reflected` | Every Reactive tick |
| L2 | Reflection and feedback history | `strategy_changed` / `strategy_expired` | Scheduler-compatible v1 |
| L3 | Procedure, reflection, strategy, belief, counterpart, feedback history | `self_model_updated` only | Manual/scheduler-compatible v1 with 12h emit throttle |

L3 does not directly write beliefs, strategies, or lenses. `SubjectProjection`
materializes `self_model_updated` into `subject_view`, so later Reactive ticks
see the updated `Subject.self_model`.

## L1 Rules

- `low-confidence-high-stakes`: a judgment has confidence below `0.4` and an
  `existence` or safety-aligned weight above `0.7`.
- `contradiction-accepted`: a judgment lists the same belief as both support
  and contradiction (`undermined_by` in the current model).
- `situation-mismatch`: a judgment declares applicability for a specific
  situation that is not the current perception situation.
- `unsupported-tool-call`: a `use_tool` decision has no judgment text requiring
  tool use.
- `premature-novel-auto-form`: a novel interpretation with confidence below
  `0.5` reports newly affected belief ids.
- `feedback-surprise`: feedback failed to match expectation and recorded at
  least one surprise.

## L3 Aggregators

- `capabilities_self_assessed`: active learned procedures by confidence,
  success count, and failure count.
- `typical_failure_modes`: reflection kind frequency.
- `preferred_strategies`: strategy references emitted by `reflector_l2`.
- `stable_preferences`: high-confidence active value beliefs about the subject,
  not about a specific counterpart.
- `typical_value_tradeoffs`: `belief_superseded` events with
  `decisive_value_kinds`.
- `interaction_patterns_by_counterpart_role`: role-level tick, feedback, and
  reflection patterns. This summarizes agent behavior by role, not individual
  counterpart beliefs.
