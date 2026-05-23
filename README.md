# Alpha Agent

Alpha Agent is a personal agent runtime with explicit memory layers inspired by
human cognition. The first version is intentionally small and controllable: it
runs from the CLI, stores experience in SQLite, retrieves memory without
embeddings, builds a transparent prompt, and uses either a mock LLM or an
OpenAI-compatible chat completions provider.

This is not a LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, or similar
framework wrapper. The goal is to own the execution flow directly.

## Relationship to Hermes Agent

Alpha Agent uses Hermes Agent as a practical reference for product usability,
especially around model-provider wiring, gateway operation, Feishu/WeChat
access, session routing, status reporting, and local operations. The intent is
usability parity where it matters for daily use, not internal design parity.

The core agent runtime remains Alpha's own design: explicit turn execution,
SQLite-backed memory layers, deterministic retrieval, salience scoring, and
manual consolidation. Hermes' plugin/provider/gateway implementation is treated
as reference material for integration decisions, not as code to copy wholesale.
See `docs/TODO.md` for the current Hermes-informed roadmap.

## What Human-Like Memory Means Here

Human-like memory in this project means separating memory by role instead of
putting everything into one transcript:

- Session context: append-only conversation messages plus optional compressed
  summaries for long-running sessions.
- Episodic memory: specific experiences and remembered events.
- Semantic memory: durable facts, preferences, and user-specific knowledge.
- Procedural memory: reusable ways of doing things, stored as skills.
- Salience scoring: deterministic importance estimates for what should persist.
- Consolidation: a manual pass that promotes stable facts from episodes.

The implementation is transparent and basic. It is designed to be inspected,
changed, and extended.

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

Inspect memory:

```bash
uv run alpha memory list
uv run alpha memory search "sqlite preferences"
uv run alpha memory stats
uv run alpha memory consolidate
```

Inspect procedural skills:

```bash
uv run alpha skills list
```

Print the prompt without calling the LLM:

```bash
uv run alpha debug prompt "what do you remember about my preferences?"
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
`codex.access_token`, `memory.retrieval_limit`, and `context.max_prompt_tokens`.
Secret values are masked by `alpha config get` unless you pass
`--reveal-secret`.

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

[memory]
retrieval_limit = 8

[context]
max_prompt_tokens = 6000
compression_threshold_ratio = 0.85
recent_tail_messages = 8
min_summary_tokens = 256
max_summary_tokens = 1024
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
- `ALPHA_RETRIEVAL_LIMIT`: Retrieval limit per memory layer. Defaults to `8`.
- `ALPHA_CONTEXT_MAX_PROMPT_TOKENS`: Prompt budget before compression.
- `ALPHA_CONTEXT_COMPRESSION_THRESHOLD_RATIO`: Ratio of the prompt budget that
  triggers compression.
- `ALPHA_CONTEXT_RECENT_TAIL_MESSAGES`: Uncompressed transcript tail to preserve.
- `ALPHA_CONTEXT_MIN_SUMMARY_TOKENS`: Lower target for compressed summaries.
- `ALPHA_CONTEXT_MAX_SUMMARY_TOKENS`: Upper target for compressed summaries.

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

## Retrieval

This version deliberately avoids embeddings. Retrieval uses:

- SQLite FTS5 when available.
- LIKE-based fallback search when FTS5 is unavailable.
- Keyword overlap.
- Recency.
- Salience.
- Access count.
- Memory type boost.
- Lightweight entity hints from title-cased names.

The ranking formula is explicit:

```text
score =
  keyword_score * 0.40
  + salience * 0.25
  + recency_score * 0.20
  + access_score * 0.10
  + type_boost * 0.05
```

## Current Limitations

- No vector retrieval yet.
- No web UI.
- No background scheduler.
- No multi-agent system.
- No real Feishu or WeChat adapter yet.
- Memory extraction is deterministic and basic.
- Graph memory is minimal.

## Roadmap

1. LLM-assisted memory extraction with review.
2. Vector retrieval as an optional module.
3. Richer graph consolidation.
4. Background dreaming/consolidation.
5. Tool execution system.
6. Local files / notes ingestion.
7. API server.
8. Web UI.
9. Channel integrations.

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
