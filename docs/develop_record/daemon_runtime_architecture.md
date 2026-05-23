# Daemon Runtime Architecture Plan

## Status
Active design record.

Date: 2026-05-23

## Objective
Alpha Agent should have exactly one long-running runtime owner process. Local CLI
commands and external gateway adapters should enter the same daemon instead of
creating separate in-process runtimes.

The daemon owns:

- IPC server.
- `AgentManager`.
- Per-session `AlphaAgent` instances.
- `GatewayRuntimeBridge`.
- Configured `PlatformAdapter` connections.
- SQLite-backed `MemoryStore` use for live runtime turns and gateway adapter
  execution.
- Status, pid, logs, and lifecycle cleanup.

The CLI owns local user interaction, IPC client behavior, and narrow
admin/diagnostic SQLite commands that do not create an `AlphaAgent` or execute a
runtime turn.

## Target Architecture
```text
alpha daemon start
  ├─ IPC server
  ├─ AgentManager
  │    └─ session_id/session_key -> AlphaAgent
  ├─ GatewayRuntimeBridge
  ├─ configured PlatformAdapters
  ├─ MemoryStore / SQLite
  └─ status / pid / logs

alpha ask/chat
  └─ IPC client -> daemon

gateway adapters
  └─ inbound message -> daemon-owned GatewayRuntimeBridge
```

## Key Decisions
- The daemon is the only runtime owner. Do not introduce a second long-running
  gateway process.
- Daemon ownership is scoped to live runtime turns and gateway adapter
  execution. Local admin and diagnostic commands may open `MemoryStore` directly
  when they only inspect or explicitly manage SQLite state.
- `AlphaAgent` is not a global singleton. `AgentManager` caches one agent per
  active `session_id` so mutable runtime state is not shared between sessions.
- The daemon owns one shared active-turn guard for all input channels. `ask`,
  `chat`, and gateway turns for the same `session_id` must not run concurrently.
- `ask` and `chat` must not call `_build_agent()` directly. They should submit
  requests over local IPC.
- Gateway adapters are daemon-owned input channels. They connect during daemon
  startup and remain connected for the daemon lifetime.
- Diagnostic retrieval is read-only. Commands such as `memory search` and
  `debug prompt` may rank stored memories locally, but they must not write
  memory access logs or increment access counters.
- `skills list` is read-only and lists stored procedural memories only. Commands
  with explicit management intent, such as `init` and `gateway doctor`, may load
  builtin skills into storage.
- If no gateway adapters are configured, the daemon still runs and serves local
  `ask` and `chat`.
- Local chat and gateway share the same runtime session system. Their source
  context differs, but both resolve to daemon-managed `session_id` values.
- Prefer direct refactoring to the target architecture. Avoid maintaining a
  parallel CLI runtime path and daemon runtime path.

## Module Layout
```text
src/alpha_agent/daemon/
  models.py      # IPC request/response/status models.
  server.py      # Unix socket JSON-lines server.
  client.py      # CLI IPC client.
  manager.py     # AgentManager: per-session AlphaAgent cache.
  status.py      # pid/socket/status helpers.
  runtime.py     # AlphaDaemon lifecycle.

src/alpha_agent/gateway/
  runner.py      # GatewayRuntimeBridge remains, but daemon owns it.
```

## IPC Contract
Use Unix domain sockets with JSON lines. Do not introduce an HTTP framework for
the first implementation.

Minimum request types:

```json
{"type": "ask", "message": "hello", "session_id": null}
{"type": "chat_turn", "message": "hello", "session_id": "s1"}
{"type": "consolidate_memory"}
{"type": "status"}
{"type": "stop"}
```

Minimum response types:

```json
{"ok": true, "session_id": "s1", "response": "..."}
{"ok": true, "status": {"state": "running"}}
{"ok": false, "error": {"code": "DAEMON_NOT_RUNNING", "message": "Daemon is not running. Run alpha daemon start."}}
```

Protocol rules:

- One JSON object per line.
- Every response includes `ok`.
- Error responses use stable machine-readable `error.code`.
- Request validation happens at the IPC boundary.
- Unknown request types return `UNKNOWN_REQUEST_TYPE`.
- Daemon unavailable is reported by the client as `DAEMON_NOT_RUNNING`.
- Local request source metadata is explicit, even if it is only `{"channel":
  "cli", "command": "ask"}` in the first pass.

## Current Code Alignment
The current code already has useful building blocks, but the plan needs these
implementation constraints to avoid false starts:

- `src/alpha_agent/cli.py` currently owns `_provider()`, `_store()`, and
  `_build_agent()`. Move that construction into daemon-owned factory code before
  migrating `ask` and `chat`.
- `ask`, `chat`, and `gateway run` currently build or use an in-process
  `AlphaAgent`. These call sites are the main migration targets.
- `GatewayRuntimeBridge` currently accepts one concrete `AlphaAgent`. Change it
  to resolve the mapped `session_id` through `AgentManager` for every non-cached
  runtime turn.
- `ActiveTurnGuard` currently protects only gateway turns. Move the guard to the
  daemon runtime path so all local and gateway input channels share the same
  per-session admission control.
- `MemoryStore` is a lightweight SQLite connection factory. The daemon can own
  one initialized `MemoryStore` and inject it into all per-session agents, while
  SQLite write serialization continues to rely on the existing immediate
  transactions.
- `conversation_messages.source_metadata` exists, but `AlphaAgent.respond()`
  currently accepts only `user_message` and `session_id`. Add a turn input or
  optional source metadata argument before claiming local/gateway source context
  is recorded.
- Existing gateway status helpers are gateway-scoped. Replace or subsume them
  with daemon status helpers instead of creating a separate status system.
- Current config has `gateway_status_path` but no daemon socket path or daemon
  status path. The daemon slice must add explicit runtime paths or derive them
  from existing runtime configuration.
- `chat` currently implements `/consolidate` locally. Migrating `chat` to IPC
  should either add a daemon `consolidate_memory` request or deliberately remove
  that in-chat command. The implementation plan chooses a daemon request.
- Local diagnostic commands still use `_store()` for SQLite inspection and
  management. This is acceptable only outside live `ask`, `chat`, and gateway
  turn execution, and diagnostic retrieval must opt out of access recording.

## AgentManager
Responsibilities:

```text
get_or_create(session_id) -> AlphaAgent
evict_idle()
evict_all()
cancel(session_id)
```

Initial cache policy:

- Key: `session_id`.
- Value: `AlphaAgent`.
- Idle TTL: 1 hour.
- Max size: 128 active agents.
- Each agent uses the daemon-owned initialized `MemoryStore`.
- Each agent owns independent mutable runtime helpers.
- Provide a release hook even if current `AlphaAgent` has no resource cleanup API.

`AgentManager` should own the reusable agent factory currently embedded in CLI
helper code. CLI code should not construct providers, retrievers, or agents after
the daemon client migration.

## CLI Command Surface
Target commands:

```text
alpha daemon start
alpha daemon run
alpha daemon status
alpha daemon stop

alpha ask "..."
alpha chat --session s1

alpha gateway doctor
alpha gateway status
```

Command behavior:

- `alpha daemon start`: starts the single runtime owner process in the
  background, including the IPC server and configured gateway adapters.
- `alpha daemon run`: runs the same daemon owner in the foreground for debugging
  and process supervisors.
- `alpha daemon status`: reads daemon status over IPC when available, with file
  status as fallback diagnostics.
- `alpha daemon stop`: requests graceful daemon shutdown over IPC.
- `alpha ask`: sends one request to daemon. If daemon is unavailable, fail with
  `Daemon is not running. Run alpha daemon start.`
- `alpha chat`: keeps the local interactive prompt, but each turn is an IPC
  `chat_turn` request.
- `alpha gateway doctor`: remains a local diagnostic command for database,
  logs, config, adapter availability, and builtin skill initialization.
- `alpha gateway status`: becomes a gateway-focused view of daemon status.
- `alpha gateway run`: should not create a second runtime. Remove it in the
  direct refactor.

Read-only local commands:

- `alpha memory list`
- `alpha memory search`
- `alpha memory stats`
- `alpha skills list`
- `alpha debug prompt`

Local management commands:

- `alpha init`
- `alpha config ...`
- `alpha memory review`
- `alpha memory consolidate`
- `alpha gateway doctor`

None of these local commands weakens the requirement that `alpha ask`,
`alpha chat`, and gateway adapter turns enter the daemon over IPC.

## Gateway Integration
`GatewayRuntimeBridge` should resolve agents lazily through `AgentManager`
instead of holding one fixed `AlphaAgent`.

Target shape:

```python
mapping = session_store.get_or_create(source, session_mode)
agent = agent_manager.get_or_create(mapping.session_id)
result = agent.respond(message.text, session_id=mapping.session_id)
```

The final runtime call should carry source metadata:

```python
result = agent.respond(
    message.text,
    session_id=mapping.session_id,
    source_metadata=gateway_source_metadata(message.source),
)
```

Gateway responsibilities remain:

- Normalize inbound platform messages.
- Deduplicate inbound delivery.
- Map external source context to daemon session ids.
- Guard active turns per session.
- Send outbound responses.
- Cache outbound retry data when delivery fails.

The daemon owns adapter lifecycle:

- Instantiate configured adapters during startup.
- Connect adapters after IPC/runtime initialization succeeds.
- Disconnect adapters during graceful stop and startup failure cleanup.
- Keep daemon running even when no adapters are configured.

## Implementation Slices
### Slice 1: Daemon IPC and Status
Acceptance criteria:

- `alpha daemon start` starts a background daemon with a Unix socket JSON-lines server.
- `alpha daemon run` runs the same Unix socket JSON-lines server in the foreground.
- `alpha daemon status` reports running state through IPC.
- `alpha daemon stop` requests shutdown.
- Status file includes pid, socket path, state, updated timestamp, and adapter names.
- No `AlphaAgent` is created yet.
- `alpha gateway run` does not start another runtime after daemon commands exist;
  it should point users to `alpha daemon start` or be removed in the same refactor.

Likely files:

- `src/alpha_agent/daemon/models.py`
- `src/alpha_agent/daemon/server.py`
- `src/alpha_agent/daemon/client.py`
- `src/alpha_agent/daemon/status.py`
- `src/alpha_agent/cli.py`
- `tests/test_daemon_*.py`

### Slice 2: AgentManager and `ask`
Acceptance criteria:

- Daemon can handle an `ask` request.
- `AgentManager.get_or_create(session_id)` creates and caches agents by session.
- `alpha ask` no longer calls `_build_agent()` directly.
- Missing daemon returns `DAEMON_NOT_RUNNING`.
- Local `ask` requests pass source metadata into the runtime.
- Daemon uses shared active-turn guard for the requested session.
- Focused tests cover request handling without requiring a true background process.

Likely files:

- `src/alpha_agent/daemon/manager.py`
- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/cli.py`
- `tests/test_daemon_runtime.py`
- `tests/test_cli_daemon.py`

### Slice 3: `chat` as IPC Client
Acceptance criteria:

- `alpha chat --session s1` sends every turn as `chat_turn`.
- The daemon response controls the displayed `session_id` and assistant text.
- Local `/exit` and `/quit` remain client-side.
- `/consolidate` sends a daemon `consolidate_memory` request and does not access
  local runtime objects directly.
- Chat turns pass source metadata into the runtime.

Likely files:

- `src/alpha_agent/cli.py`
- `src/alpha_agent/daemon/client.py`
- `src/alpha_agent/daemon/models.py`
- `tests/test_cli_daemon.py`

### Slice 4: Gateway Adapters Inside Daemon
Acceptance criteria:

- Daemon startup connects configured adapters.
- Adapter inbound messages enter daemon-owned `GatewayRuntimeBridge`.
- `GatewayRuntimeBridge` asks `AgentManager` for the mapped session agent.
- Gateway turns pass platform source metadata into the runtime.
- Gateway and local IPC turns share daemon active-turn admission by `session_id`.
- No configured adapters still yields a healthy daemon.
- `alpha gateway run` no longer starts an independent runtime.

Likely files:

- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/gateway/runner.py`
- `src/alpha_agent/cli.py`
- `tests/test_daemon_gateway.py`
- `tests/test_gateway_core.py`

### Slice 5: Lifecycle Hardening
Acceptance criteria:

- Pid lock prevents double daemon startup.
- Signal handling cleans up socket/status and disconnects adapters.
- Stop policy is explicit: graceful drain or immediate stop.
- Agent cache eviction works by idle TTL and max size.
- Runtime status distinguishes `starting`, `running`, `stopping`, `idle`, and
  `error`.

Likely files:

- `src/alpha_agent/daemon/status.py`
- `src/alpha_agent/daemon/runtime.py`
- `src/alpha_agent/daemon/manager.py`
- `tests/test_daemon_lifecycle.py`

## Risks
- Unix socket availability differs by platform. This project currently targets
  local development usage, so Unix socket IPC is acceptable for the first pass.
- A synchronous socket server can block if runtime turns are long. The first
  implementation should use one thread per client request or a small executor.
- `AlphaAgent` currently lacks an explicit release API. `AgentManager` should
  provide a release path now and call into agent cleanup later.
- Moving `chat` to daemon IPC requires `/consolidate` to become a daemon request;
  otherwise chat would keep a hidden local runtime path.
- Gateway delivery retry state must remain internal and must not trust adapter
  raw metadata.

## Verification Plan
- Unit tests for IPC model parsing and error responses.
- Integration-style tests for daemon request handler with fake agent manager.
- CLI tests for daemon unavailable behavior.
- Gateway bridge tests proving per-session agent resolution through manager.
- Full suite after each slice: `PYTHONPATH=. uv run pytest -q`.
- Lint changed files after each slice: `uv run ruff check ...`.

## Open Questions
- Should `alpha ask` create a new session every call, or should it optionally
  accept `--session` for continuity?
- Should `ask` and `chat` auto-start the daemon, or should explicit `alpha daemon start`
  remain mandatory?
- Should session cache eviction be time-based only, or also persist recent usage
  metadata for diagnostics?
