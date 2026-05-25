# Cognition Reflectors

Phase 05 implements L1 as a deterministic, read-only audit over one completed
Reactive tick. L1 emits reflections for inspection only; it does not mutate
beliefs, strategy, subject state, or the next-turn policy.

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
