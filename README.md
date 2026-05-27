# Alpha Agent

Alpha Agent is a personal agent runtime for rebuilding cognition from first
principles. The current baseline is intentionally small and controllable: it
runs from the CLI, stores session-level and cognition state in SQLite, drives
successful turns through a Reactive cognition tick, and uses either a mock LLM
or an OpenAI-compatible chat completions provider.

This is not a LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, or similar
framework wrapper. The goal is to own the execution flow directly.

## Relationship to Hermes Agent

Alpha Agent uses Hermes Agent as a practical reference for product usability,
especially around model-provider wiring, gateway operation, Feishu/WeChat
access, session routing, status reporting, and local operations. The intent is
usability parity where it matters for daily use, not internal design parity.

The core agent runtime remains Alpha's own design: explicit turn execution,
SQLite-backed state, Reactive stage orchestration, and direct provider/tool
wiring. Hermes' plugin/provider/gateway implementation is treated as reference
material for integration decisions, not as code to copy wholesale.
See `docs/todo/TODO.md` for the current Hermes-informed roadmap.

## Cognition Status

Phase 00 cleared the previous long-term record subsystem. Phase 01 cognition
foundations are now in place: typed models, the cognitive event log, projection
infrastructure, the counterpart materialized view, and the single-subject
LoopCoordinator.

Phase 02 Reactive tick is now wired into `AlphaAgent.respond()`. A successful
user turn flows through Perceive, Attend, Interpret, Judge, Decide, Act,
Feedback, Reflect, and Revise, with a shared `tick_id` in the cognitive event
log. The Reactive loop uses non-blocking acquisition: when the single-subject
coordinator is busy, `respond()` returns a busy result immediately, does not
preempt the current holder, and does not write cognitive events or conversation
messages for the rejected stimulus.

The default Reactive Effector executes a bounded tool loop itself: one tool
iteration followed by a final LLM round. Phase 09 renderer extraction is now in
place: Effector receives a `CognitionView`, calls `TextChatRenderer` by default,
and feeds rendered messages into the existing LLM/tool loop. The old
`runtime/prompt_builder.py` path has been removed.

Phase 03 BeliefProjection is complete: beliefs are materialized in SQLite and
recallable across sessions through a deterministic projection over cognition
events. Phase 04 ContextWindowProjection is also complete for foreground
context: `context_window_view` stores thread-local foreground perception IDs,
anchors, and rebuildable window state. Belief recall is joined into
`ContextWindow.recalled` during the Reactive tick. Phase 05 L1 reflection is
also complete: every tick emits a `reflected` event, with rule findings
materialized into `reflection_view`. Phase 06 deterministic consolidation v1 is
available through a synchronous `run_once`: it merges equivalent beliefs,
archives expired beliefs, learns minimal procedures, compresses foreground
context into background summaries, and maintains counterpart digest beliefs.
Phase 07 deterministic ValueLens v1 is also in place: subject lenses persist in
SQLite, queued conflicts resolve through lens-shaped scoring, ties are kept for
human review, and a conservative consolidation worker can nudge lens
sensitivity from repeated resolved tradeoffs. Phase 08 Reflector L2
deterministic control v1 adds temporary strategy overrides:
`strategy_view` stores active controls, L2 rules can emit `strategy_changed`,
expired strategies are cleared by consolidation, and Reactive stages honor the
implemented strategy names. Phase 10 Drive Loop v1 adds event-sourced goals in
`goal_view` and a disabled-by-default synchronous manual loop that turns one
eligible active goal into a cognition-thread `self_signal`. Phase 11 Reflector
L3 v1 deterministically aggregates long-window cognition history into
`Subject.self_model` and materializes it in `subject_view`.

## Install

```bash
uv sync
```

Initialize the local SQLite database:

```bash
uv run alpha init
```

Create or inspect the local config file:

```bash
uv run alpha config init
uv run alpha config show
uv run alpha config set llm.provider codex
uv run alpha config get llm.provider
```

## CLI Usage

Start the daemon runtime owner before local turns or gateway adapters:

```bash
uv run alpha daemon start
uv run alpha daemon status
uv run alpha daemon stop
```

For foreground debugging or process supervisors, run the daemon without
backgrounding:

```bash
uv run alpha daemon run
```

Start an interactive chat:

```bash
uv run alpha chat
```

Run a single turn:

```bash
uv run alpha ask "hello"
```

Inspect procedural skills:

```bash
uv run alpha skills list
```

Print a renderer prompt preview without calling the LLM:

```bash
uv run alpha debug prompt "summarize the current session" --renderer text_chat
uv run alpha debug prompt "summarize this channel" --session <session-id>
```

Inspect cognition renderer outputs:

```bash
uv run alpha cognition graph --format mermaid
uv run alpha cognition diff <tick-id-a> <tick-id-b>
uv run alpha cognition evidence <belief-id>
uv run alpha cognition consolidate --now --dry-run
uv run alpha cognition lens show
uv run alpha cognition lens set --priority safety,honesty,efficiency
uv run alpha cognition strategies --active
uv run alpha cognition strategy-expire <strategy-id>
uv run alpha cognition goals set --description "answer pending question" --priority 5
uv run alpha cognition goals list --active
uv run alpha cognition goals satisfy <goal-id> --evidence "accepted"
uv run alpha cognition goals abandon <goal-id> --reason "obsolete"
uv run alpha cognition drive --once
uv run alpha cognition self-model
uv run alpha cognition self-model history --last 5
uv run alpha cognition reflect-l3 --once
```

Inspect raw LLM request/response traces from CLI runs:

```bash
uv run alpha config set llm.debug_logging true
tail -f ~/.alpha-agent/logs/llm.jsonl
```

SQLite `runtime_traces` keep LLM summaries and correlation ids; full LLM
request/response payloads are written only to the JSONL debug log when
`llm.debug_logging` is enabled.

Inspect the gateway operational shell:

```bash
uv run alpha gateway doctor
uv run alpha gateway status
```

Gateway adapters are owned by `alpha daemon start`. `alpha gateway doctor` remains
a local diagnostic command for SQLite gateway tables, log files, and adapter
availability.

## Configuration

Alpha reads long-lived settings from `~/.alpha-agent/config.toml` by default.
Run `uv run alpha config init` to create it. You can also point to another file
with `ALPHA_CONFIG_PATH`.

Use `alpha config set <section.key> <value>` for supported keys such as
`llm.provider`, `llm.model`, `llm.debug_logging`, `deepseek.api_key`,
`codex.access_token`, `llm.context.expected_output_reserve_tokens`, and
`llm.providers.deepseek.max_context_tokens`.
Secret values are masked by `alpha config get` unless you pass
`--reveal-secret`.
Config loading applies the same validation as `alpha config set`: token and
count limits must be positive integers, and context threshold ratios must be in
the `(0, 1]` range.

Environment variables and `.env` still work as overrides for one-off runs,
deployment, and secrets. Precedence is:

```text
defaults < config.toml < .env / environment variables
```

Main config keys:

```toml
[runtime]
db_path = "~/.alpha-agent/alpha.db"
log_dir = "~/.alpha-agent/logs"
gateway_status_path = "~/.alpha-agent/gateway-status.json"
daemon_socket_path = "~/.alpha-agent/daemon.sock"
daemon_status_path = "~/.alpha-agent/daemon-status.json"

[llm]
provider = "mock"
model = "" # empty means "use the selected provider's default"

[llm.context]
tool_truncate_threshold_ratio = 0.60
handover_compress_threshold_ratio = 0.90
minimum_remaining_tokens = 10000
tool_string_truncate_chars = 300
expected_output_reserve_tokens = 4096
safety_margin_tokens = 1024

[llm.providers.openai-compatible]
max_context_tokens = 258400

[llm.providers.deepseek]
max_context_tokens = 1000000

[compatible]
base_url = "https://api.openai.com/v1"
api_key = ""

[deepseek]
api_key = ""
reasoning_enabled = true

[codex]
access_token = ""

[cognition.drive]
enabled = false
interval_seconds = 300
goal_cooldown_seconds = 3600
active_goal_limit = 8
```

Useful environment overrides:

- `ALPHA_CONFIG_PATH`: Config file path. Defaults to
  `~/.alpha-agent/config.toml`.
- `ALPHA_DB_PATH`: SQLite database path. Defaults to `~/.alpha-agent/alpha.db`.
- `ALPHA_LOG_DIR`: Gateway log directory. Defaults to `~/.alpha-agent/logs`.
- `ALPHA_GATEWAY_STATUS_PATH`: Gateway status JSON path. Defaults to
  `~/.alpha-agent/gateway-status.json`.
- `ALPHA_DAEMON_SOCKET_PATH`: Daemon Unix socket path. Defaults to
  `~/.alpha-agent/daemon.sock`.
- `ALPHA_DAEMON_STATUS_PATH`: Daemon status JSON path. Defaults to
  `~/.alpha-agent/daemon-status.json`.
- `ALPHA_LLM_PROVIDER`: `mock`, `openai-compatible`, `deepseek`, or `codex`.
  Defaults to `mock`.
- `ALPHA_LLM_MODEL`: Optional model override for the selected provider. Empty
  uses the provider default.
- `ALPHA_LLM_DEBUG_LOGGING`: Set to `true` to write full LLM request/response
  traces. Defaults to `false`.
- `ALPHA_COMPATIBLE_BASE_URL`: Base URL for a chat completions compatible API.
- `ALPHA_COMPATIBLE_API_KEY`: API key for the compatible provider.
- `ALPHA_DEEPSEEK_API_KEY`: DeepSeek API key when `ALPHA_LLM_PROVIDER=deepseek`.
- `ALPHA_CODEX_ACCESS_TOKEN`: Optional Codex OAuth bearer token. If omitted,
  Alpha tries `CODEX_HOME/auth.json` or `~/.codex/auth.json`.
- `ALPHA_LLM_CONTEXT_TOOL_TRUNCATE_THRESHOLD_RATIO`: Tool replay payload
  truncation threshold. Defaults to `0.60`.
- `ALPHA_LLM_CONTEXT_HANDOVER_COMPRESS_THRESHOLD_RATIO`: LLM handover
  compression threshold. Defaults to `0.90`.
- `ALPHA_LLM_CONTEXT_MINIMUM_REMAINING_TOKENS`: Minimum remaining context budget.
  Defaults to `10000`.
- `ALPHA_LLM_CONTEXT_TOOL_STRING_TRUNCATE_CHARS`: Tool input/output string
  truncation length. Defaults to `300`.
- `ALPHA_LLM_CONTEXT_EXPECTED_OUTPUT_RESERVE_TOKENS`: Reserved output budget.
  Defaults to `4096`.
- `ALPHA_LLM_CONTEXT_SAFETY_MARGIN_TOKENS`: Context estimate safety margin.
  Defaults to `1024`.
- `ALPHA_COGNITION_DRIVE_ENABLED`: Enables scheduled Drive Loop use when a
  caller wires it in. Defaults to `false`.
- `ALPHA_COGNITION_DRIVE_INTERVAL_SECONDS`: Global Drive Loop interval setting.
- `ALPHA_COGNITION_DRIVE_GOAL_COOLDOWN_SECONDS`: Per-goal self-signal cooldown.
- `ALPHA_COGNITION_DRIVE_ACTIVE_GOAL_LIMIT`: Maximum concurrently active goals.

The mock provider works without an API key:

```bash
ALPHA_LLM_PROVIDER=mock uv run alpha ask "hello"
```

DeepSeek and Codex provider examples:

```bash
ALPHA_LLM_PROVIDER=deepseek ALPHA_DEEPSEEK_API_KEY=... uv run alpha ask "hello"
ALPHA_LLM_PROVIDER=codex uv run alpha ask "hello"
```

Codex uses OAuth-style bearer credentials. The simplest path is to log in with
Codex CLI first so `~/.codex/auth.json` exists; `ALPHA_CODEX_ACCESS_TOKEN` is
only an override.

## Runtime State

The current SQLite state baseline is deliberately narrow:

- `session_messages`: append-only source stream for user, assistant, tool, and
  compressed handover messages used to assemble LLM-visible session context.
- `runtime_traces`: operational turn, provider, and tool traces.
- `gateway_session_mappings`: platform/session routing state.
- `gateway_dedup`: inbound gateway deduplication state.
- `cognitive_events`: Phase 01 append-only cognition event log.
- `counterpart_view`: Phase 01 materialized view for counterpart projection
  queries.
- `belief_view`: Phase 03 materialized view for active, superseded, and
  retracted beliefs, with deterministic recall across sessions.
- `context_window_view`: Phase 04 materialized view for thread-local foreground
  ContextWindow state, including perception IDs and anchors.
- `context_window_background`: Phase 06 deterministic background summaries for
  compressed foreground context.
- `reflection_view`: Phase 05 materialized view for L1 reflection findings.
- `procedure_view`: Phase 06 minimal learned procedure projection.
- `strategy_view`: Phase 08 temporary strategy overrides emitted by L2 or
  manual expiry.
- `goal_view`: Phase 10 active/satisfied/abandoned goal materialization for the
  Drive Loop.
- `subject_view`: Phase 11 current Subject materialization, including
  `SelfModel`.
- `cognition_worker_checkpoint`: Phase 06 consolidation worker progress.
- `subject_value_lens`: Phase 07 current subject ValueLens priority and
  sensitivity.

Successful user turns now enter the Reactive tick before producing a response.
BeliefProjection, ContextWindowProjection with background compression,
ReflectionProjection, ProcedureProjection, and renderer-driven prompt assembly
are now in place. Subject ValueLens persistence, deterministic conflict
resolution, queued conflict consumption, and temporary strategy overrides are
also in place. GoalProjection and manual DriveLoop self-signals are in place.
SubjectProjection now persists the L3 SelfModel from `self_model_updated`.
Semantic strategy/lens diff remains pending.

## Current Limitations

- Consolidation is deterministic v1 only: no background daemon scheduler, no LLM
  summarization policy, and no daemon-owned worker cadence.
- ValueLens v1 is deterministic: it uses explicit `ValueKind` weights and
  sensitivity, not semantic moral reasoning or full adaptive learning.
- Reflector L2 v1 is deterministic: no strategy DSL, no semantic clustering,
  and no daemon-owned L2 scheduler. It provides scheduler-compatible work units
  and CLI inspection/expiry.
- Drive Loop v1 is synchronous and disabled by default: no daemon-owned drive
  cadence, no autonomous goal generation, and one self-signal per manual pass.
- Reflector L3 v1 is deterministic: no LLM self-narration, no direct
  belief/strategy/lens writes, and no daemon-owned L3 cadence.
- Semantic strategy/lens diff is still pending.
- No web UI.
- No multi-agent system.
- No real Feishu or WeChat adapter yet.

## Roadmap

1. Cognition runtime Phase 01: event log foundations. Completed.
2. Cognition runtime Phase 02: reactive loop and counterpart routing. Completed.
3. Cognition runtime Phase 03: belief projection. Completed.
4. Cognition runtime Phase 04: foreground context window. Completed.
5. Cognition runtime Phase 05: Reflector L1. Completed.
6. Cognition runtime Phase 09: renderer extraction. Completed.
7. Cognition runtime Phase 06: deterministic consolidation loop v1. Completed.
8. Cognition runtime Phase 07: deterministic ValueLens conflict resolution v1.
   Completed.
9. Cognition runtime Phase 08: deterministic Reflector L2 control v1.
   Completed.
10. Cognition runtime Phase 10: goal projection and synchronous Drive Loop v1.
    Completed.
11. Cognition runtime Phase 11: deterministic Reflector L3 SelfModel v1.
    Completed.
12. Cognition runtime Phase 09+: semantic strategy/lens diff.
13. Tool execution system.
14. Local files / notes ingestion.
15. API server.
16. Web UI.
17. Channel integrations.

## Development

Run tests:

```bash
uv run pytest
```

Run linting and type checks:

```bash
uv run ruff check .
uv run mypy src tests
```
