# Cognition Reflectors

The foreground runtime owns user and self-signal turns. Reflectors consume
durable audit/projection data after those turns complete; they do not recreate
the old foreground stage pipeline.

L2 adds deterministic strategy control. L3 adds deterministic SelfModel
aggregation.

| Level | Reads | Writes | Cadence |
| --- | --- | --- | --- |
| L2 | Reflection, value-lens, and belief lifecycle history | `strategy_changed` / `strategy_expired` | Scheduler-compatible v1 |
| L3 | Procedure, reflection, strategy, belief, counterpart, feedback history | `self_model_updated` only | Manual/scheduler-compatible v1 with 12h emit throttle |

L3 does not directly write beliefs, strategies, or lenses. `SubjectProjection`
materializes `self_model_updated` into `subject_view`, so later runtime turns
can inspect the updated `Subject.self_model`.

## L3 Aggregators

- `capabilities_self_assessed`: active learned procedures by confidence,
  success count, and failure count.
- `typical_failure_modes`: reflection kind frequency.
- `preferred_strategies`: strategy references emitted by `reflector_l2`.
- `stable_preferences`: high-confidence active value beliefs about the subject,
  not about a specific counterpart.
- `typical_value_tradeoffs`: `belief_superseded` events with
  `decisive_value_kinds`.
- `interaction_patterns_by_counterpart_role`: role-level turn, feedback, and
  reflection patterns. This summarizes agent behavior by role, not individual
  counterpart beliefs.
