# Phase 09 Renderer Completion Record

## Status

Completed.

## Delivered

- Added `cognition/render/` with `CognitionView`, `Renderer`,
  `RenderBudget`, `RenderResult`, and `build_view`.
- Added four built-in renderers:
  - `TextChatRenderer` for chat-completions messages.
  - `GraphSnapshotRenderer` for Mermaid/DOT belief graph snapshots.
  - `DiffRenderer` for deterministic tick-to-tick event-kind diffs.
  - `EvidenceRenderer` for belief lifecycle evidence chains.
- Refactored Effector so it receives a `CognitionView`, calls the injected
  renderer, and passes rendered messages into the existing LLM/tool loop.
- Preserved runtime tool-loop behavior: renderer builds the initial prompt,
  runtime still appends assistant tool calls and tool results.
- Removed `src/alpha_agent/runtime/prompt_builder.py` and all `src/` imports.
- Migrated prompt-builder tests into renderer tests and retained the
  session-context tail/tool-pair test separately.
- Added CLI renderer paths:
  - `alpha debug prompt --renderer text_chat`
  - `alpha cognition graph --format mermaid|dot`
  - `alpha cognition diff <tick_id_a> <tick_id_b>`
  - `alpha cognition evidence <belief_id>`

## Implementation Choices

- `TextChatRenderer` is the default Effector renderer. It is counterpart-aware:
  role selects system prompt template, communication style becomes a prompt
  segment, and low trust marks recalled beliefs as user-reported/unverified.
- `DiffRenderer` intentionally operates on event kinds available today. It
  lists belief/value-lens/strategy event deltas, but does not claim semantic
  strategy or lens diff until those projections exist.
- `EvidenceRenderer` reads the append-only event log and emits lifecycle event
  inputs/outputs. Perception-level traceability appears when events carry
  perception inputs.
- `CognitionView.chat_history` exists for debug prompt previews and does not
  reintroduce the old prompt-builder layer.

## Verification

- Targeted renderer/runtime tests passed during implementation:
  `25 passed`.
- Final required verification:
  - `uv run ruff check .` passed.
  - `uv run pytest -q` passed: `205 passed`.
- `rg -n "prompt_builder" src/alpha_agent` returned no matches.

## Follow-Up

- Populate `counterpart_digest` after the consolidation/digest projection
  exists.
- Upgrade `DiffRenderer` from event-kind diff to semantic lens/strategy diff
  after Phase 07/08/11 materialize those concepts.
- Add provider-specific renderers, such as Anthropic block rendering, in a
  separate provider phase.
