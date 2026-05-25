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
iteration followed by a final LLM round. `AlphaAgent.respond()` does not
pre-build LLM input through `SessionContextManager` + `PromptBuilder`; it
injects a runtime runner at the Effector boundary so successful turns still
persist transcript and tool traces. Renderer extraction remains deferred to
Phase 09. `alpha debug prompt --trace` prints the baseline prompt preview plus
the recent cognitive event chain for the session; it is not the final Phase 09
renderer output.

Phase 03 BeliefProjection is complete: beliefs are materialized in SQLite and
recallable across sessions through a deterministic projection over cognition
events. Phase 04 ContextWindowProjection is also complete for foreground
context: `context_window_view` stores thread-local foreground perception IDs,
anchors, and rebuildable window state. Belief recall is joined into
`ContextWindow.recalled` during the Reactive tick. Procedure projection remains
stubbed, and reflection, consolidation, value lens, renderer extraction, and
drive loop remain staged under `docs/todo/cognition-runtime/`.

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

Print a baseline prompt preview without calling the LLM:

```bash
uv run alpha debug prompt "summarize the current session"
uv run alpha debug prompt "summarize this channel" --session <session-id>
```

Inspect raw LLM request/response traces from CLI runs:

```bash
uv run alpha config set llm.debug_logging true
tail -f ~/.alpha-agent/logs/llm.jsonl
```

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
`codex.access_token`, `context.max_prompt_tokens`, and
`context.recent_tail_messages`.
Secret values are masked by `alpha config get` unless you pass
`--reveal-secret`.
Config loading applies the same validation as `alpha config set`: token and
count limits must be positive integers.

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

[compatible]
base_url = "https://api.openai.com/v1"
api_key = ""

[deepseek]
api_key = ""
reasoning_enabled = true

[codex]
access_token = ""

[context]
max_prompt_tokens = 6000
recent_tail_messages = 8
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
- `ALPHA_CONTEXT_MAX_PROMPT_TOKENS`: Prompt budget for the current turn.
- `ALPHA_CONTEXT_RECENT_TAIL_MESSAGES`: Uncompressed transcript tail to preserve.

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

- `conversation_messages`: append-only session transcript used for current
  successful turn context.
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

Successful user turns now enter the Reactive tick before producing a response.
BeliefProjection and foreground ContextWindowProjection are now real
SQLite-backed projections. Procedure projection, background compression, and
renderer-driven prompt assembly remain pending.

## Current Limitations

- Procedure projection, reflection, consolidation, value lens, renderer
  extraction, and drive loop are still pending.
- No web UI.
- No multi-agent system.
- No real Feishu or WeChat adapter yet.

## Roadmap

1. Cognition runtime Phase 01: event log foundations. Completed.
2. Cognition runtime Phase 02: reactive loop and counterpart routing. Completed.
3. Cognition runtime Phase 03: belief projection. Completed.
4. Cognition runtime Phase 04: foreground context window. Completed.
5. Cognition runtime Phase 05+: reflection, consolidation, value lens, renderer
   extraction, and drive loop.
6. Tool execution system.
7. Local files / notes ingestion.
8. API server.
9. Web UI.
10. Channel integrations.

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
