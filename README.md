# Alpha Agent

Alpha Agent is a personal agent runtime for rebuilding cognition from first
principles. The current baseline is intentionally small and controllable: it
runs from the CLI, stores session-level and cognition state in SQLite, drives
successful turns through a daemon-owned LLM/tool loop, and uses either a mock
LLM or a configured provider (`openai-compatible`, `deepseek`, or `codex`).
Local `ask` and `chat` turns are daemon-owned, and the runtime has an explicit
bounded LLM/tool loop with an optional Tavily-backed `web_search` tool.

This is not a LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, or similar
framework wrapper. The goal is to own the execution flow directly.

## Relationship to Hermes Agent

Alpha Agent uses Hermes Agent as a practical reference for product usability,
especially around model-provider wiring, gateway operation, Feishu/WeChat
access, session routing, status reporting, and local operations. The intent is
usability parity where it matters for daily use, not internal design parity.

The core agent runtime remains Alpha's own design: explicit turn execution,
SQLite-backed state, turn-owned cognition audit/projection hooks, and direct
provider/tool wiring. Hermes' plugin/provider/gateway implementation is treated
as reference material for integration decisions, not as code to copy wholesale.
See `docs/todo/TODO.md` for the current Hermes-informed roadmap.

## Cognition Status

The previous long-term record subsystem has been removed. The current cognition
runtime is built around typed models, an append-only cognitive event log,
SQLite-backed projections, a counterpart materialized view, and a
single-subject `LoopCoordinator`.

User-originated turns route through a stable default counterpart,
`counterpart:main-user`. Local CLI turns without platform identity use that
counterpart directly, and the first platform user observed through a gateway
claims the same counterpart. Later distinct platform users receive their own
counterpart identities.

`AlphaAgent.respond()` owns the foreground turn. A successful user turn
allocates one runtime turn identity, persists the user message, runs the bounded
provider/tool loop, persists assistant and tool messages, and records cognition
audit events that projections and workers can materialize later. The foreground
path uses non-blocking acquisition: when the single-subject coordinator is busy,
`respond()` returns a busy result immediately, does not preempt the current
holder, and does not write cognitive events or conversation messages for the
rejected stimulus.

Key write-side cognition events are validated before append for the payload
fields consumed by projections and consolidation workers. Successful
`AlphaAgent.respond()` turns also record session-source linkage: `perceived`
events carry the user source message id, and a `turn_sources_recorded` event
links the runtime turn to the persisted assistant, provider tool messages, and
runtime trace ids.

The foreground runtime assembles prompt messages from session history, stable
counterpart profile snapshots, and compact context maintenance, then feeds them
into AlphaAgent's LLM/tool loop. In daemon-owned turns, provider-returned tool
calls are persisted as assistant `tool_calls` and matching tool messages,
executed by `ToolExecutor`, bounded by `max_tool_iterations` and
`max_llm_rounds`, and finalized with `tool_choice=none` when a limit is reached.
The old `runtime/prompt_builder.py` path has been removed.

Stable counterpart profile context is selected from counterpart digest beliefs
once per ordinary session and rendered near the start of the prompt when
available. It is a compact, always-visible, session-stable digest snapshot, not
a dynamic search result.

Dynamic memory recall is available only through an explicit `memory_recall` tool
call during the normal provider tool loop. Runtime passes the recall context to
the tool executor, but it does not decide when recall is needed and does not
automatically inject recalled beliefs into the prompt. Tool-visible recall
results remain compact belief content without ids, confidence scores, sources,
or evidence.

Memory writes use the separate `memory_propose` proposal path for explicit
long-term preferences, constraints, procedures, and corrections during a
runtime turn. Accepted low-risk proposals emit `memory_proposed`, then
`belief_formed` or `belief_superseded`, and apply immediately to `belief_view`;
pending or rejected proposals remain audit-only.

Beliefs are materialized in SQLite and recallable across sessions through a
deterministic projection over cognition events. Foreground context is stored in
`context_window_view` as session-scoped perception IDs, anchors, and rebuildable
window state. Model-facing dynamic lookup goes through `memory_recall`, while
stable profile context and compact session context are assembled before the
provider call. Concrete audit findings are materialized into `reflection_view`.

Deterministic consolidation is available through a synchronous `run_once`: it
merges equivalent beliefs, archives expired beliefs, learns minimal procedures,
compresses foreground context into background summaries, maintains counterpart
digest beliefs, resolves queued conflicts, learns conservative ValueLens
sensitivity shifts, expires strategies, and can run the deterministic L3
self-model worker. Subject lenses persist in SQLite; queued conflicts resolve
through lens-shaped scoring, ties are kept for human review, and a conservative
consolidation worker can nudge lens sensitivity from repeated resolved
tradeoffs.

Temporary strategy overrides are stored in `strategy_view`: L2 rules can emit
`strategy_changed`, expired strategies are cleared by consolidation, and the
prompt renderer surfaces implemented strategy reminders to the foreground
runtime. The Drive Loop stores event-sourced goals in `goal_view` and exposes a
disabled-by-default synchronous manual loop that can enqueue one eligible active
goal as a self-signal. The L3 reflector deterministically aggregates long-window
cognition history into `Subject.self_model` and materializes it in
`subject_view`.

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
uv run alpha daemon restart
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
uv run alpha chat --session <session-id>
```

Passing `--session` resumes that session and prints a compact preview of recent
user/assistant messages before accepting the next turn.
When a chat turn uses tools, the CLI renders the current turn's assistant
tool-call messages and tool results in order before the final answer.

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
uv run alpha debug prompt "summarize this channel" --session <session-id> --trace
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
uv run alpha cognition strategies --all
uv run alpha cognition strategy-expire <strategy-id>
uv run alpha cognition reflections --last 20
uv run alpha cognition reflections --severity warning
uv run alpha cognition goals set --description "answer pending question" --priority 5
uv run alpha cognition goals set --description "answer pending question" --target-outcome "accepted answer"
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
`runtime.db_path`, `runtime.log_dir`, `llm.provider`, `llm.model`,
`llm.debug_logging`, `compatible.base_url`, `compatible.api_key`,
`deepseek.api_key`, `deepseek.reasoning_enabled`,
`deepseek.reasoning_effort`, `codex.access_token`, `tavily.api_key`,
`tools.bash.enabled`, `tools.bash.default_workdir`,
`tools.bash.allowed_workdirs`, `tools.bash.default_timeout_seconds`,
`tools.bash.max_timeout_seconds`, `tools.bash.max_output_chars`,
`tools.bash.env_passthrough`, `llm.context.expected_output_reserve_tokens`, and
`llm.providers.deepseek.max_context_tokens`. Drive Loop settings are also
settable through `cognition.drive.*`. Consolidation settings are loaded from
TOML and environment variables but are not currently accepted by
`alpha config get` or `alpha config set`.
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
debug_logging = false

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

[tools.bash]
enabled = false
default_workdir = "."
allowed_workdirs = ["."]
default_timeout_seconds = 120
max_timeout_seconds = 600
max_output_chars = 30000
env_passthrough = []

[cognition.consolidation]
enabled = true
interval_seconds = 300
context_foreground_max = 8
context_absorb_batch = 4
context_summary_chars = 480
counterpart_digest_min_beliefs = 5
counterpart_digest_min_new_beliefs = 3

[cognition.drive]
enabled = false
interval_seconds = 300
goal_cooldown_seconds = 3600
active_goal_limit = 8

[deepseek]
api_key = ""
reasoning_enabled = true
reasoning_effort = ""

[codex]
access_token = ""

[tavily]
api_key = ""
```

Useful environment overrides:

- `ALPHA_CONFIG_PATH`: Config file path. Defaults to
  `~/.alpha-agent/config.toml`.
- `ALPHA_DB_PATH`: SQLite database path. Defaults to `~/.alpha-agent/alpha.db`.
- `ALPHA_LOG_DIR`: Runtime log directory. Defaults to `~/.alpha-agent/logs`.
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
- `ALPHA_DEEPSEEK_REASONING_ENABLED`: Enables DeepSeek thinking parameters for
  thinking-capable models. Defaults to `true`.
- `ALPHA_DEEPSEEK_REASONING_EFFORT`: Optional DeepSeek effort override: `low`,
  `medium`, `high`, `max`, or `xhigh`.
- `ALPHA_CODEX_ACCESS_TOKEN`: Optional Codex OAuth bearer token. If omitted,
  Alpha tries `CODEX_HOME/auth.json` or `~/.codex/auth.json`.
- `ALPHA_CODEX_API_KEY`: Backward-compatible alias for
  `ALPHA_CODEX_ACCESS_TOKEN`.
- `ALPHA_TAVILY_API_KEY`: Tavily API key. Alpha also accepts `TAVILY_API_KEY`.
  When set, Alpha registers the provider-backed `web_search` tool for
  daemon-owned agent turns.
- `ALPHA_BASH_TOOL_ENABLED`: Enables the local `bash` tool. Defaults to
  `false`.
- `ALPHA_BASH_TOOL_DEFAULT_WORKDIR`: Default bash tool working directory.
  Defaults to `.`.
- `ALPHA_BASH_TOOL_ALLOWED_WORKDIRS`: Comma-separated workdir allowlist for the
  bash tool. Defaults to `.`.
- `ALPHA_BASH_TOOL_DEFAULT_TIMEOUT_SECONDS`: Default foreground command timeout.
  Defaults to `120`.
- `ALPHA_BASH_TOOL_MAX_TIMEOUT_SECONDS`: Maximum accepted foreground command
  timeout. Defaults to `600`.
- `ALPHA_BASH_TOOL_MAX_OUTPUT_CHARS`: Maximum bash stdout/stderr content
  returned to the model after cleanup. Defaults to `30000`.
- `ALPHA_BASH_TOOL_ENV_PASSTHROUGH`: Comma-separated environment variable names
  to pass through to bash. Defaults to empty; provider secrets are still blocked.
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
- `ALPHA_COGNITION_CONSOLIDATION_ENABLED`: Enables manual consolidation passes.
  Defaults to `true`.
- `ALPHA_COGNITION_CONSOLIDATION_INTERVAL_SECONDS`: Worker schedule interval
  setting; manual `--now` runs force a pass immediately.
- `ALPHA_COGNITION_CONSOLIDATION_JUDGMENT_REPEAT_WINDOW`: Recent judgment
  window inspected by repeat-detection workers.
- `ALPHA_COGNITION_CONSOLIDATION_JUDGMENT_REPEAT_THRESHOLD`: Repetition count
  needed before judgment promotion.
- `ALPHA_COGNITION_CONSOLIDATION_PROCEDURE_SUCCESS_THRESHOLD`: Success count
  needed before procedure learning.
- `ALPHA_COGNITION_CONSOLIDATION_CONTEXT_FOREGROUND_MAX`: Foreground context
  count before background compression is eligible.
- `ALPHA_COGNITION_CONSOLIDATION_CONTEXT_ABSORB_BATCH`: Number of foreground
  perceptions absorbed per compression worker pass.
- `ALPHA_COGNITION_CONSOLIDATION_CONTEXT_SUMMARY_CHARS`: Deterministic
  background summary length.
- `ALPHA_COGNITION_CONSOLIDATION_COUNTERPART_DIGEST_MIN_BELIEFS`: Minimum
  active beliefs before counterpart digest summarization.
- `ALPHA_COGNITION_CONSOLIDATION_COUNTERPART_DIGEST_MIN_NEW_BELIEFS`: Minimum
  new beliefs before refreshing a counterpart digest.
- `ALPHA_COGNITION_DRIVE_ENABLED`: Enables scheduled Drive Loop use when a
  caller wires it in. Defaults to `false`.
- `ALPHA_COGNITION_DRIVE_INTERVAL_SECONDS`: Global Drive Loop interval setting.
- `ALPHA_COGNITION_DRIVE_GOAL_COOLDOWN_SECONDS`: Per-goal self-signal cooldown.
- `ALPHA_COGNITION_DRIVE_ACTIVE_GOAL_LIMIT`: Maximum concurrently active goals.

The daemon reads provider settings when it starts. After changing provider or
tool credentials in config, restart the daemon before running `ask` or `chat`.
The mock provider works without an API key:

```bash
uv run alpha config set llm.provider mock
uv run alpha daemon restart
uv run alpha ask "hello"
```

DeepSeek and Codex provider examples:

```bash
uv run alpha config set llm.provider deepseek
uv run alpha config set deepseek.api_key ...
uv run alpha daemon restart
uv run alpha ask "hello"

uv run alpha config set llm.provider codex
uv run alpha daemon restart
uv run alpha ask "hello"
```

Codex uses OAuth-style bearer credentials. The simplest path is to log in with
Codex CLI first so `~/.codex/auth.json` exists; `ALPHA_CODEX_ACCESS_TOKEN` is
only an override. Environment overrides still work, but they must be present in
the daemon process environment, not only on the client-side `ask` command.

The built-in `web_search` tool is available when `tavily.api_key`,
`ALPHA_TAVILY_API_KEY`, or `TAVILY_API_KEY` is configured:

```bash
uv run alpha config set tavily.api_key tvly-...
uv run alpha daemon restart
```

The built-in `bash` tool is disabled by default. Enable it only for trusted
local use when you want the agent to run foreground build, test, package, Git,
diagnostic, or system-inspection commands:

```bash
uv run alpha config set tools.bash.enabled true
uv run alpha config set tools.bash.allowed_workdirs ".,~/alpha-work"
uv run alpha daemon restart
```

`bash` is not a security sandbox. The configured workdir allowlist only controls
the process `cwd`; it is not filesystem confinement, so commands can still refer
to absolute paths that the daemon user can access. The tool uses a cleaned
environment, timeout/cancel handling, dangerous-command blocking, and output
redaction/truncation. It does not allow background process escape hatches such
as `nohup`, `disown`, `setsid`, or a trailing `&`, and it blocks privileged or
interactive commands such as `sudo` and `vim`. Do not expose it to untrusted
remote gateway users without a stronger sandbox or approval layer.

## Runtime State

The current SQLite state baseline is deliberately narrow:

- `session_messages`: append-only source stream for user, assistant, tool, and
  compressed handover messages used to assemble LLM-visible session context,
  including assistant `reasoning_content` when a provider supplies it.
- `session_counterparts`: first counterpart identity bound to each session.
- `session_profile_snapshots`: session-stable counterpart profile snapshots
  keyed by session.
- `runtime_traces`: operational turn, provider, and tool traces.
- `gateway_session_mappings`: platform/session routing state.
- `gateway_dedup`: inbound gateway deduplication state.
- `cognitive_events`: append-only cognition event log.
- `counterpart_view`: materialized view for counterpart projection queries.
- `belief_view`: materialized view for active, superseded, and retracted
  beliefs, with deterministic recall across sessions.
- `belief_entity_index` and `belief_about_index`: lookup indexes for belief
  entity/about references.
- `context_window_view`: materialized view for session-scoped foreground
  ContextWindow state, including perception IDs and anchors.
- `context_window_background`: deterministic background summaries for compressed
  foreground context.
- `reflection_view`: materialized view for L1 reflection findings.
- `procedure_view`: minimal learned procedure projection.
- `strategy_view`: temporary strategy overrides emitted by L2 or manual expiry.
- `goal_view`: active/satisfied/abandoned goal materialization for the Drive
  Loop.
- `subject_view`: current Subject materialization, including `SelfModel`.
- `cognition_worker_checkpoint`: consolidation worker progress.
- `subject_value_lens`: current subject ValueLens priority and sensitivity.

Successful user turns now run through the foreground LLM/tool loop before
cognition audit records and projections are finalized. BeliefProjection,
ContextWindowProjection with background compression, ReflectionProjection,
ProcedureProjection, and renderer-driven prompt assembly are now in place.
Subject ValueLens persistence, deterministic conflict resolution, queued
conflict consumption, and temporary strategy overrides are also in place.
GoalProjection and manual DriveLoop self-signals are in place. SubjectProjection
now persists the L3 SelfModel from `self_model_updated`. The explicit tool
execution subsystem is also in place, with `memory_propose` and `memory_recall`
always registered, `web_search` registered when Tavily credentials are
configured, and `bash` registered only when `tools.bash.enabled=true`.
Semantic strategy/lens diff remains pending.

## Current Limitations

- Consolidation is deterministic v1 only: it runs through manual synchronous
  CLI passes, not a daemon-owned cadence, and it has no LLM summarization
  policy.
- ValueLens v1 is deterministic: it uses explicit `ValueKind` weights and
  sensitivity, not semantic moral reasoning or full adaptive learning.
- Reflector L2 v1 is deterministic: no strategy DSL, no semantic clustering,
  and no daemon-owned L2 scheduler. It provides scheduler-compatible work units
  and CLI inspection/expiry.
- Drive Loop v1 is synchronous and disabled by default: no daemon-owned drive
  cadence, no autonomous goal generation, and one self-signal per manual pass.
- Reflector L3 v1 is deterministic: it can run manually or through a manual
  consolidation pass, but has no LLM self-narration, no direct
  belief/strategy/lens writes, and no daemon-owned L3 cadence.
- Semantic strategy/lens diff is still pending.
- No web UI.
- No multi-agent system.
- No real Feishu or WeChat adapter yet.

## Roadmap

1. Semantic strategy/lens diff.
2. Local files / notes ingestion.
3. API server.
4. Web UI.
5. Channel integrations.

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
