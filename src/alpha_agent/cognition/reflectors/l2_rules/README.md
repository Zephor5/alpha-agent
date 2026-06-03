# Reflector L2 Rules v1

Deterministic Phase 08 rules emit temporary `StrategyOverride` records. They do
not emit belief events or change the ValueLens directly.

| Rule | Trigger | Strategy |
| --- | --- | --- |
| `recurring-contradiction-accepted` | Three same-kind contradiction reflections within 30 minutes. | `require_explicit_confirm_on_contradiction` |
| `lens-shift-flap` | Three `value_lens_shifted` events with the same deterministic direction key in 24 hours. Direction is derived from before/after priority or sensitivity deltas, with trigger text only as fallback. | `freeze_lens_learning_for_24h` |

The v1 rule set is intentionally shallow: no free-form DSL, no semantic
clustering, and no self-modifying L2 behavior.
