# Simplify Reactive Cognition Pipeline

## Status
Proposed

## Date
2026-06-02

## Objective
Remove the current reactive cognition pipeline as a behavior-driving path. Keep cognition useful by narrowing it to durable responsibilities: turn audit, projections, memory/tool state transitions, and background consolidation.

The target architecture should not let deterministic string rules pretend to understand natural language. Natural-language semantics should stay with the main LLM turn and with tools that own state changes, especially memory write tools.

The foreground execution identity is the runtime agent turn. `turn_id` must be allocated once by the agent runtime and carried through LLM calls, tool calls, memory writes, cognition audit events, persisted session messages, and runtime traces. New foreground code should not introduce a second behavior identity such as `tick_id`.

## Core Decision
The standalone `reactive_tick` should not remain a separate primary execution path after the stage pipeline is removed.

Reason: once `Interpreter`, `Judger`, `Decider`, and `Reviser` stop making behavior decisions, `reactive_tick` mostly becomes a wrapper around event emission plus the final LLM call. That duplicates the runtime agent loop without adding clear value.

Target replacement:

```text
main runtime turn
  -> allocate one AgentTurnContext with turn_id
  -> assemble prompt/context
  -> main LLM decides response and tool calls
  -> tools perform constrained state changes
  -> runtime-owned audit hooks record useful cognition events
  -> cognition projections materialize durable views
  -> background workers consolidate durable state
```

## Target Contracts

### Agent Turn Identity

`turn_id` is the foreground causal identity. It belongs to the runtime agent loop, not to cognition stages, tools, projections, or background workers.

`session_id` is the foreground transcript and context bucket. It owns persisted session messages, tool messages, runtime traces, prompt history, context-window foreground state, and self-signal streams. The new runtime turn contract does not include `thread_id`; existing `ThreadId` usage is migration debt to remove.

Proposed internal shape:

```python
@dataclass(frozen=True)
class AgentTurnContext:
    turn_id: str
    session_id: str
    started_at: Instant
    source: Reference | None
    counterpart: Reference | None
    user_message_id: str | None = None
    turn_received_event_id: str | None = None
```

Rules:
- Allocate `turn_id` once at the accepted runtime turn boundary.
- Pass `AgentTurnContext` explicitly into the model/tool loop and tool extension context.
- Use `turn_id` in new cognition event payloads, runtime traces, tool result metadata, and source-linking payloads.
- Use `session_id` as the context-window key for foreground runtime context.
- Remove active `thread_id`/`ThreadId` dependencies from the foreground runtime path instead of deriving a new thread id from session state.
- Delete `tick_id` from the next-stage foreground path; active foreground code must not produce, consume, or validate it.
- Do not use debug dictionaries as the source of truth for causal identity.
- Busy or rejected attempts are not completed foreground turns unless the runtime explicitly records them as separate rejection audit records.

### Runtime Audit Interface

The runtime audit interface records boundaries; it must not interpret user intent or decide behavior.

Required operations:
- `record_turn_received`: records the inbound user message or self-signal as auditable foreground input.
- `record_turn_acted`: records the completed model/tool loop outcome, including tool-call and tool-result references.
- `record_turn_sources`: links persisted session messages, runtime traces, tool messages, and cognition events for one turn.

Every emitted event from this interface must follow the foreground payload contracts below. The interface should be small and explicit, replacing the nine-stage controller as the foreground event write path.

### Foreground Event Vocabulary

| Event kind | Target foreground decision |
| --- | --- |
| `PERCEIVED` | Keep as runtime turn input audit, with `turn_id` and persisted source refs. |
| `ATTENDED` | Remove as foreground stage output unless a concrete projection still needs a non-semantic focus event. |
| `INTERPRETED` | Remove as foreground natural-language semantic output. |
| `JUDGED` | Remove as generic per-message judgment output. |
| `DECIDED` | Remove from active foreground events; actual model/tool decisions are represented by persisted LLM/tool messages, runtime traces, and `ACTED`/`TURN_SOURCES_RECORDED`. |
| `ACTED` | Keep or replace as runtime model/tool-loop outcome audit, with `turn_id`. |
| `RECEIVED_FEEDBACK` | Do not emit automatic placeholder feedback for every turn; reserve for real external/service feedback or explicitly documented background signals. |
| `REFLECTED` | Keep only for audits based on concrete runtime/tool outcomes; remove rules that require obsolete interpretation/judgment inputs. |
| `REVISED` | Remove placeholder foreground revision events. |
| `MEMORY_PROPOSED` | Keep, but payload and metadata must use `turn_id` and tool-call audit refs. |
| `BELIEF_FORM_PENDING_CONFIRMATION` | Keep only if tied to a concrete memory proposal or policy gate. |
| `TURN_SOURCES_RECORDED` | Keep, but source and payload should reference `turn_id`, not a cognition tick. |

### Foreground Payload Contracts

These contracts describe only the next-stage active shape. Do not add `schema_version`, `thread_id`, `tick_id`, `decision_id`, `judgment_ids`, or compatibility fields for the old stage pipeline.

Common required fields for retained foreground audit events:

```json
{
  "turn_id": "turn:...",
  "session_id": "..."
}
```

`PERCEIVED` records accepted runtime turn input. It is the source for foreground context-window state and input audit.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "stimulus_kind": "user_message|self_signal",
  "source": {"kind": "session", "id": "session-1"},
  "from_counterpart": {"kind": "counterpart", "id": "..."},
  "source_refs": [
    {"kind": "session", "id": "session-1"},
    {"kind": "session_message", "id": "..."}
  ],
  "content_digest": "...",
  "content_length": 123
}
```

`ACTED` records the actual model/tool-loop outcome. It must not depend on a pre-LLM `Decision`.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "assistant_message_id": "...",
  "response_text_digest": "...",
  "response_text_length": 123,
  "llm_call_ids": ["llm:..."],
  "llm_trace_ids": ["trace:..."],
  "tool_call_ids": ["call:..."],
  "tool_names": ["memory_propose"],
  "tool_result_trace_ids": ["trace:..."],
  "tool_cognitive_event_ids": ["event:..."]
}
```

`MEMORY_PROPOSED` records the proposed memory state transition. Raw proposal content belongs here because this event is the audit record for the proposed state change, even though the default user-facing response stays memory-silent.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "proposal_id": "proposal:...",
  "tool_call_id": "call:...",
  "proposal": {
    "kind": "preference|constraint|correction|procedure",
    "scope": "counterpart|global",
    "content": "...",
    "evidence": "..."
  },
  "derived_about": [{"kind": "counterpart", "id": "..."}],
  "gate": {
    "decision": "accepted|pending_confirmation|rejected",
    "reason": "..."
  },
  "source_refs": [],
  "audit_refs": []
}
```

`BELIEF_FORM_PENDING_CONFIRMATION` records an explicit user-action requirement for one concrete memory proposal.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "proposal_id": "proposal:...",
  "reason": "correction_requires_review|ambiguous_conflict",
  "required_user_action": "confirm_memory_change",
  "candidate_change": {
    "kind": "create|replace|correct",
    "content": "..."
  },
  "conflict_belief_ids": ["belief:..."]
}
```

`TURN_SOURCES_RECORDED` is the join point for persisted artifacts created during one turn.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "user_message_id": "...",
  "assistant_message_id": "...",
  "provider_tool_message_ids": [],
  "provider_tool_trace_ids": [],
  "llm_call_ids": [],
  "llm_trace_ids": [],
  "cognitive_event_ids": [],
  "tool_cognitive_event_ids": []
}
```

`REFLECTED` is retained only if concrete audit findings remain after obsolete interpretation/judgment inputs are removed.

```json
{
  "turn_id": "turn:...",
  "session_id": "session-1",
  "reflection_count": 1,
  "reflection_ids": ["reflection:..."],
  "targets": [
    {"kind": "tool_call", "id": "call:..."}
  ]
}
```

### Memory Proposal Result Contract

`memory_propose` should return structured results so the model loop can decide whether the user must be involved. Memory and cognition should stay mostly invisible: successful memory writes should not trigger default user-facing announcements.

Target result shape:

```json
{
  "status": "accepted|pending_confirmation|rejected|mixed",
  "user_action": "none|ask_confirmation|explain_rejection",
  "message_hint": "",
  "proposal_results": [
    {
      "proposal_id": "proposal:...",
      "decision": "accepted|pending_confirmation|rejected",
      "reason": "..."
    }
  ]
}
```

Rules:
- `accepted` may emit `BELIEF_FORMED` or `BELIEF_SUPERSEDED`.
- `accepted` defaults to `user_action: "none"`; the main LLM should not announce memory saves unless the user explicitly asked for confirmation of what was remembered.
- `pending_confirmation` must not silently mutate memory.
- `pending_confirmation` uses `user_action: "ask_confirmation"` and should ask only about the concrete proposed state change that needs approval.
- `rejected` uses `user_action: "explain_rejection"` only when the user explicitly asked the agent to remember something and the tool rejects it; ordinary rejected proposals can remain invisible.
- For `mixed` results, derive one `user_action` by priority: `ask_confirmation` first, then `explain_rejection`, then `none`.
- Detailed audit fields such as belief ids, superseded belief ids, conflict ids, and cognition event ids belong in tool metadata and cognition events. They do not need to be part of the default user-visible tool output.
- Conflict detection must use structured proposal fields and targeted belief retrieval. It must not recreate a hidden generic semantic analyzer.

## What Moves Where

| Current responsibility | Target owner | Rationale |
| --- | --- | --- |
| Allocate foreground execution identity | Runtime agent turn | `turn_id` is shared by LLM rounds, tools, audit events, traces, and persisted messages. |
| Own foreground context bucket | Runtime session | `session_id` owns transcript history, prompt context, context-window state, and self-signal streams. |
| Understand user intent | Main LLM | The main model already sees the full conversation and decides tool calls. |
| Decide whether and how to recall memory | Main LLM invokes `memory_recall`; recall tool scopes explicit query | Runtime must not run implicit recall from the current user message. |
| Decide whether to remember something | Main LLM proposes; memory tool gates | Memory write is a state transition and belongs in the memory tool path. |
| Decide what memory record to create | Memory tool builder and validators | The write path must normalize, validate, scope, and persist records. |
| Detect memory conflict | Memory tool write path | Conflict is part of write consistency, not a generic interpretation stage. |
| Require confirmation before memory mutation | Memory tool or policy gate | Confirmation should be tied to the exact proposed state change. |
| Decide whether to call tools | Main LLM tool calling plus tool policy | A separate string-rule decider is redundant. |
| Record foreground audit events | Runtime audit interface | Event emission should mirror actual turn boundaries, not invented stage boundaries. |
| Track projections | Cognition projection layer | This remains valuable and deterministic. |
| Merge, archive, summarize, resolve durable state | Background cognition workers | These are asynchronous maintenance jobs, not foreground semantic stages. |
| Drive self-signals | Runtime turn producer with internal session ids | Drive creates normal runtime turns using stable internal session ids such as goal-scoped sessions. |

## Non-Goals
- Do not add a new unified semantic analyzer as another LLM call in each turn.
- Do not improve the current string-based `Interpreter` rules.
- Do not keep compatibility with the old reactive stage behavior.
- Do not preserve `reactive_tick` as a public behavior path unless a concrete active caller still requires it during migration.
- Do not introduce a parallel foreground identity beside runtime `turn_id`.
- Do not keep `thread_id` as a foreground runtime identity; session id owns the context bucket.
- Do not preserve `tick_id` in active audit payloads, tool contexts, or payload validators.
- Do not perform implicit memory recall from the current user message in the foreground runtime path.

## Work Plan

### Phase 0: Establish Runtime Turn Contract

**Task 0.1: Define one `AgentTurnContext` allocation path.**

Acceptance criteria:
- [ ] `turn_id` is allocated exactly once at the accepted runtime turn boundary.
- [ ] `AgentTurnContext` carries `turn_id`, `session_id`, `started_at`, source, counterpart, persisted user message id, and optional received-event id.
- [ ] Runtime methods pass `AgentTurnContext` explicitly instead of reconstructing causal identity from debug fields.
- [ ] New foreground runtime code uses `session_id` as the context bucket and does not create or pass `thread_id`.
- [ ] New foreground code does not produce, consume, or validate `tick_id`.
- [ ] Busy/rejected attempts do not produce completed-turn audit records.

Verification:
- [ ] Add tests proving one user turn has one stable `turn_id` across model calls, tool calls, traces, and returned debug metadata.
- [ ] Add tests proving foreground context-window state is keyed by `session_id`, not `thread_id`.
- [ ] Run targeted runtime turn tests.

Likely touched:
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/runtime/`
- `tests/test_agent_loop.py`

**Task 0.2: Add a runtime-owned cognition audit interface.**

Acceptance criteria:
- [ ] Runtime can record turn input, model/tool-loop outcome, and persisted source links without constructing `CognitiveController`.
- [ ] Runtime audit events contain `turn_id`, persisted source references, and causal parent event ids where applicable.
- [ ] Audit hooks do not interpret natural language, judge claims, choose tools, or derive revisions.

Verification:
- [ ] Add tests for turn-received, turn-acted, and turn-sources-recorded event payloads.
- [ ] Run cognition payload contract tests and runtime event tests.

Likely touched:
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/emitter.py`
- `src/alpha_agent/cognition/payload_contract.py`
- `tests/`

**Task 0.3: Lock the foreground event vocabulary before deleting stages.**

Acceptance criteria:
- [ ] Each event kind in the foreground event vocabulary table above has a documented action: keep, replace, reserve, or remove.
- [ ] Consumers of `JUDGED`, `DECIDED`, and `RECEIVED_FEEDBACK` have explicit replacement sources or removal actions.
- [ ] Payload contracts define only the next-stage active foreground shape and reject `schema_version`, `thread_id`, `tick_id`, `decision_id`, and `judgment_ids`.

Verification:
- [ ] Run `rg -n "tick_id|PERCEIVED|ATTENDED|INTERPRETED|JUDGED|DECIDED|ACTED|RECEIVED_FEEDBACK|REFLECTED|REVISED|TURN_SOURCES_RECORDED" src tests`.
- [ ] Add or update event payload contract tests for the new turn-owned event shape.

Likely touched:
- `src/alpha_agent/cognition/models/enums.py`
- `src/alpha_agent/cognition/payload_contract.py`
- `src/alpha_agent/cognition/projections/`
- `src/alpha_agent/cognition/loops/workers/`
- `tests/cognition/`

**Task 0.4: Move tool extension identity from tick context to turn context.**

Acceptance criteria:
- [ ] Tool extension context for memory write and recall receives `AgentTurnContext` or equivalent structured turn data.
- [ ] `memory_propose` causal parents refer to the runtime turn input/audit event and tool-call refs, not a synthetic decision event.
- [ ] Tool results expose `status` and `user_action` so the model loop can avoid mentioning memory side effects, ask confirmation, or explain rejection deterministically.

Verification:
- [ ] Add tests for tool extension context shape and memory proposal causal refs.
- [ ] Run targeted memory tool and runtime tool-loop tests.

Likely touched:
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/tools/memory_recall.py`
- `tests/cognition/test_memory_propose_tool.py`
- `tests/test_agent_loop.py`

**Task 0.5: Migrate DriveLoop/self-signal behavior to internal sessions.**

Acceptance criteria:
- [ ] DriveLoop produces normal runtime turns with `turn_id`.
- [ ] DriveLoop uses stable internal `session_id` values for goal-scoped self-signal streams.
- [ ] DriveLoop no longer calls `reactive_tick`; it uses the same turn contract as user-message turns.
- [ ] Goal progress and cooldown semantics remain tied to concrete runtime turn completion.

Verification:
- [ ] Run DriveLoop and CLI goal tests.
- [ ] Run `rg -n "DriveLoop|self_signal|reactive_tick|ThreadId|thread_id" src tests docs README.md`.

Likely touched:
- `src/alpha_agent/cognition/loops/drive.py`
- `src/alpha_agent/cli.py`
- `src/alpha_agent/runtime/agent.py`
- `tests/cognition/test_drive_loop_behavior.py`
- `tests/cognition/test_cli_goals.py`

Checkpoint:
- [ ] `turn_id` is the only foreground causal identity in new runtime-owned code.
- [ ] `session_id` is the only foreground context bucket in new runtime-owned code.
- [ ] Runtime can emit useful audit events without the nine-stage controller.
- [ ] Memory tools can receive turn identity without depending on `Decision`, `tick_id`, or `thread_id`.
- [ ] DriveLoop has an internal-session migration path before `reactive_tick` removal begins.

### Phase 1: Map Active Usage

**Task 1: Identify all callers of `CognitiveController.reactive_tick`.**

Acceptance criteria:
- [ ] Every direct caller is listed.
- [ ] Each caller is classified as runtime behavior, DriveLoop/self-signal behavior, CLI/debug behavior, or test-only behavior.
- [ ] A removal or migration action is assigned to each caller.
- [ ] No active caller is allowed to keep `reactive_tick` as a primary behavior path.

Verification:
- [ ] Run `rg -n "reactive_tick|CognitiveController" src tests`.

Likely touched:
- `src/alpha_agent/cognition/controller.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/drive.py`
- `src/alpha_agent/cli.py`
- `tests/`

**Task 2: Identify stage outputs that are still consumed outside the reactive pipeline.**

Acceptance criteria:
- [ ] Uses of `INTERPRETED`, `JUDGED`, `DECIDED`, `REVISED`, and `RECEIVED_FEEDBACK` are listed.
- [ ] Consumers of `PERCEIVED`, `ACTED`, `REFLECTED`, `MEMORY_PROPOSED`, and `TURN_SOURCES_RECORDED` that produce, consume, or validate `tick_id` are listed for deletion or rewrite.
- [ ] Active `ThreadId` and `thread_id` consumers are listed and assigned to concrete session-key migration tasks.
- [ ] Each consumer is classified as necessary, replaceable, or removable.
- [ ] Background workers depending on `JUDGED`, `DECIDED`, or automatic `RECEIVED_FEEDBACK` have replacement event sources or are scheduled for removal.

Verification:
- [ ] Run `rg -n "INTERPRETED|JUDGED|DECIDED|REVISED|RECEIVED_FEEDBACK|Judgment|Interpretation|tick_id|ThreadId|thread_id" src tests`.

Likely touched:
- `src/alpha_agent/cognition/loops/workers/`
- `src/alpha_agent/cognition/reflectors/`
- `src/alpha_agent/cognition/projections/`
- `src/alpha_agent/cognition/render/`
- `tests/cognition/`

Checkpoint:
- [ ] No code is removed before active usage is mapped.
- [ ] The replacement owner for each useful behavior is explicit.
- [ ] Event consumers are mapped from a global perspective, including runtime, DriveLoop, projections, workers, renderers, CLI, and tests.

### Phase 2: Move Runtime Turns to the Turn-Owned Audit Path

**Task 3: Route normal user turns through runtime-owned audit hooks.**

Acceptance criteria:
- [ ] A normal `Agent.respond()` turn assembles context, calls the main LLM/tool loop, writes session messages, and records cognition audit events without calling `reactive_tick`.
- [ ] The prompt/render path receives context from projections directly through runtime-owned assembly, not through `Interpreter`, `Judger`, or `Decider`.
- [ ] `TURN_SOURCES_RECORDED` links persisted user, assistant, tool, runtime trace, and cognition event refs by `turn_id`.

Verification:
- [ ] Add or update tests for runtime turn event emission.
- [ ] Run runtime agent-loop and cognition projection tests.

Likely touched:
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/render/`
- `src/alpha_agent/cognition/projections/context_window.py`
- `tests/test_agent_loop.py`
- `tests/cognition/`

**Task 4: Preserve useful context/projection behavior without semantic stages.**

Acceptance criteria:
- [ ] Context-window foreground state is populated from runtime input audit events.
- [ ] Context-window foreground state is keyed by `session_id`; active foreground code does not require `ThreadId`.
- [ ] `memory_recall` scopes explicit model-provided query parameters to the active counterpart/session context.
- [ ] Runtime does not perform implicit memory recall from the current user message during prompt assembly.
- [ ] Procedure matching is either removed from foreground auto-decision behavior or explicitly redesigned around tool/runtime outcomes.

Verification:
- [ ] Add tests for context-window foreground, counterpart-scoped explicit recall, and prompt assembly without implicit recall under the new runtime path.
- [ ] Run targeted context-window, recall, and renderer tests.

Likely touched:
- `src/alpha_agent/cognition/projections/context_window.py`
- `src/alpha_agent/cognition/projections/belief.py`
- `src/alpha_agent/cognition/render/`
- `tests/cognition/`

Checkpoint:
- [ ] Normal user turns have one semantic decision owner: the main LLM.
- [ ] New foreground audit events use `turn_id`.
- [ ] New foreground context-window access uses `session_id`.
- [ ] The runtime no longer needs `Decider` to decide whether tools are called.

### Phase 3: Strengthen Tool-Owned Memory Semantics

**Task 5: Move conflict detection into `memory_propose`.**

Acceptance criteria:
- [ ] `memory_propose` retrieves relevant existing beliefs for each accepted proposal.
- [ ] It returns structured proposal decisions: `accepted`, `pending_confirmation`, or `rejected` with concrete reasons.
- [ ] A proposal that replaces an existing belief emits the correct supersede/update event instead of creating an unrelated duplicate.
- [ ] Conflict detection uses structured proposal fields and targeted belief retrieval, not generic natural-language string comparison.

Verification:
- [ ] Add tests for no conflict, replacement conflict, ambiguous conflict, pending confirmation, and rejected proposal.
- [ ] Run targeted memory tool tests.

Likely touched:
- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/cognition/projections/belief.py`
- `tests/cognition/test_memory_propose_tool.py`

**Task 6: Make memory proposal results drive user involvement only when needed.**

Acceptance criteria:
- [ ] Tool results expose enough structured state for the main LLM to decide whether user involvement is needed.
- [ ] Prompt/tool instructions require the main LLM to avoid mentioning memory side effects for `user_action: "none"` while still answering the user normally.
- [ ] Prompt/tool instructions require the main LLM to ask for confirmation when the tool returns `user_action: "ask_confirmation"`.
- [ ] Prompt/tool instructions allow rejection explanations only when the user explicitly asked the agent to remember something.
- [ ] The tool does not silently mutate memory when confirmation is required.

Verification:
- [ ] Add tests for tool metadata and runtime prompt/tool result handling.
- [ ] Run targeted runtime and memory tests.

Likely touched:
- `src/alpha_agent/tools/memory_propose.py`
- `src/alpha_agent/runtime/agent.py`
- `tests/`

Checkpoint:
- [ ] Memory write behavior no longer depends on `Interpreter`, `Judger`, or `Reviser`.
- [ ] Memory conflicts are handled in the write path.
- [ ] Memory tool events and tool results are traceable by `turn_id`.

### Phase 4: Remove Foreground Semantic Stages

**Task 7: Remove `Interpreter` as a behavior dependency.**

Acceptance criteria:
- [ ] Foreground runtime behavior no longer calls `Interpreter`.
- [ ] String-based consistency/contradiction checks are removed or isolated to tests that are scheduled for deletion.
- [ ] Any retained interpretation event is generated only from concrete tool outcomes, not natural-language string comparison.

Verification:
- [ ] Run tests that cover runtime turns and memory proposal behavior.

Likely touched:
- `src/alpha_agent/cognition/stages/interpret.py`
- `src/alpha_agent/cognition/controller.py`
- `tests/cognition/`

**Task 8: Remove or replace `Judger`.**

Acceptance criteria:
- [ ] Runtime no longer creates generic short-lived judgments from every user message.
- [ ] Any background worker that promoted repeated judgments is removed or changed to consume explicit memory proposal or tool outcome events.
- [ ] Judgment-specific tests are deleted or rewritten around the new event source.
- [ ] Context-window recent-judgment state is removed or redesigned around a concrete replacement event.

Verification:
- [ ] Run cognition worker and context-window tests.

Likely touched:
- `src/alpha_agent/cognition/stages/judge.py`
- `src/alpha_agent/cognition/loops/workers/promote_judgment.py`
- `src/alpha_agent/cognition/projections/context_window.py`
- `tests/cognition/`

**Task 9: Remove `Decider`.**

Acceptance criteria:
- [ ] Tool selection is owned by the main LLM/tool runtime.
- [ ] The old string rule that chooses `use_tool` when the message contains `tool` is removed.
- [ ] `DECIDED` is removed from active foreground event emission; do not add another decision-shaped foreground event.
- [ ] Procedure-learning workers either consume actual runtime/tool outcome events or are removed.

Verification:
- [ ] Run runtime tool-calling tests and procedure worker tests.

Likely touched:
- `src/alpha_agent/cognition/stages/decide.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/workers/learn_procedure.py`
- `tests/`

**Task 10: Remove or narrow `Reviser`.**

Acceptance criteria:
- [ ] Memory confirmation and mutation policy lives in `memory_propose` or a dedicated memory policy gate.
- [ ] `REVISED` is no longer emitted as a placeholder event with no durable effect.
- [ ] Any remaining revision event records an actual state transition.

Verification:
- [ ] Run memory, cognition, and event payload contract tests.

Likely touched:
- `src/alpha_agent/cognition/stages/revise.py`
- `src/alpha_agent/cognition/payload_contract.py`
- `tests/cognition/`

Checkpoint:
- [ ] The foreground runtime no longer depends on `Interpreter`, `Judger`, `Decider`, or `Reviser`.
- [ ] No placeholder cognition events are emitted only to preserve the old pipeline shape.
- [ ] Workers and projections do not consume deleted stage events without an explicit replacement.

### Phase 5: Retire `reactive_tick`

**Task 11: Migrate or remove remaining `reactive_tick` callers.**

Acceptance criteria:
- [ ] Runtime user turns no longer construct `CognitiveController`.
- [ ] DriveLoop/self-signal behavior follows the Phase 0 internal-session migration decision.
- [ ] CLI/debug commands inspect actual events/projections rather than invoking the old behavior path.
- [ ] Tests that only verify the nine-stage event chain are deleted or rewritten around runtime turn behavior.

Verification:
- [ ] Run `rg -n "reactive_tick|CognitiveController" src tests`.
- [ ] Run runtime, DriveLoop/goal, CLI, and cognition projection tests.

Likely touched:
- `src/alpha_agent/cognition/controller.py`
- `src/alpha_agent/runtime/agent.py`
- `src/alpha_agent/cognition/loops/drive.py`
- `src/alpha_agent/cli.py`
- `tests/`

**Task 12: Delete or quarantine obsolete stage modules.**

Acceptance criteria:
- [ ] Obsolete stage modules are removed from imports and package exports.
- [ ] Tests that only verify old placeholder behavior are deleted.
- [ ] Remaining cognition tests verify projections, tools, workers, and audit events.

Verification:
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run mypy src tests`.
- [ ] Run `uv run pytest -q`.

Likely touched:
- `src/alpha_agent/cognition/stages/`
- `src/alpha_agent/cognition/controller.py`
- `tests/cognition/`

Checkpoint:
- [ ] `rg -n "reactive_tick|Interpreter|Judger|Decider|Reviser" src tests` has no active runtime dependency on the old pipeline.
- [ ] The project still supports memory recall, memory proposal, projections, and background consolidation.
- [ ] No active foreground code produces, consumes, or validates `tick_id`.
- [ ] No active foreground runtime path depends on `ThreadId` or `thread_id`.

### Phase 6: Documentation and Cleanup

**Task 13: Update project docs to describe the simplified architecture.**

Acceptance criteria:
- [ ] README and cognition docs describe the main LLM/tool path as the foreground behavior path.
- [ ] Docs describe `turn_id` as the runtime foreground causal identity.
- [ ] Docs describe `session_id` as the foreground transcript and context bucket.
- [ ] Docs no longer describe `thread_id` as an active foreground runtime identity.
- [ ] Docs no longer describe nine-stage reactive cognition as the current architecture.
- [ ] Memory write semantics document accepted, pending confirmation, rejected, formed, and superseded outcomes.

Verification:
- [ ] Run `rg -n "reactive tick|nine-stage|Interpreter|Judger|Decider|Reviser|tick_id|thread_id|ThreadId" README.md docs src`.

Likely touched:
- `README.md`
- `docs/cognition/`
- `src/alpha_agent/cognition/loops/README.md`

**Task 14: Remove obsolete event contracts from active code.**

Acceptance criteria:
- [ ] Event kinds that no longer occur are removed from active payload contracts.
- [ ] Renderers no longer surface obsolete foreground stages as current behavior.
- [ ] CLI/debug commands remain useful for actual event/projection inspection.
- [ ] Runtime prompts no longer refer to "reactive context" when the foreground path is turn-owned.

Verification:
- [ ] Run event payload contract tests.
- [ ] Run CLI render tests.

Likely touched:
- `src/alpha_agent/cognition/models/enums.py`
- `src/alpha_agent/cognition/payload_contract.py`
- `src/alpha_agent/cognition/render/`
- `src/alpha_agent/runtime/agent.py`
- `tests/cognition/`

Final checkpoint:
- [ ] Full validation passes:
  - [ ] `uv run ruff check .`
  - [ ] `uv run mypy src tests`
  - [ ] `uv run pytest -q`
- [ ] The foreground turn path has one semantic decision owner: the main LLM.
- [ ] The foreground turn path has one causal identity: runtime `turn_id`.
- [ ] The foreground context path has one context bucket: runtime `session_id`.
- [ ] State changes are owned by tools and policy gates.
- [ ] Cognition remains valuable as audit, projection, and background consolidation infrastructure.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `tick_id` remains as a hidden foreground causal identity | High | Establish `AgentTurnContext` first and delete `tick_id` from active foreground contracts, tool contexts, payloads, and validators. |
| `thread_id` remains as a hidden foreground context identity | High | Remove `ThreadId` from the runtime turn contract and migrate foreground context-window keys to `session_id`. |
| Removing `JUDGED` breaks background workers | Medium | Map all event consumers first and migrate workers to explicit memory/tool/runtime outcome events or remove them. |
| Removing automatic `DECIDED`/`RECEIVED_FEEDBACK` breaks procedure learning or L2/L3 aggregators | Medium | Decide whether those features consume real runtime/tool outcomes or are removed before stage deletion. |
| DriveLoop still depends on `reactive_tick` | High | Migrate DriveLoop to normal runtime turns with stable internal `session_id` values before deleting the controller path. |
| Memory write tool becomes too powerful | High | Keep strict schema validation, proposal limits, scope rules, and confirmation gates. |
| Main LLM skips memory proposal when it should remember | Medium | Improve tool instructions and add recall/proposal examples in runtime prompt. |
| Tool conflict detection becomes another weak string matcher | High | Prefer structured proposal fields and targeted belief retrieval; use LLM only inside the main turn, not as hidden extra calls. |
| Tests encode old placeholder behavior | Medium | Delete tests that only preserve obsolete pipeline shape; keep tests for user-visible behavior and durable state transitions. |

## Success Criteria
- A normal user turn does not call a standalone `reactive_tick`.
- Runtime `turn_id` is the only foreground causal identity in new runtime, tool, and audit contracts.
- Runtime `session_id` is the only foreground transcript/context bucket in new runtime and projection contracts.
- `thread_id`/`ThreadId` is removed from active foreground runtime contracts.
- No foreground behavior depends on natural-language string comparison in `Interpreter`.
- Memory creation, conflict handling, and confirmation are owned by `memory_propose` or a dedicated memory policy gate.
- Tool calling is owned by the main LLM runtime.
- DriveLoop creates normal runtime turns using stable internal `session_id` values.
- Cognition still records useful events and maintains projections without pretending to be a separate semantic reasoning engine.
