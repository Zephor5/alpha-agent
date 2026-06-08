# Alpha Agent

A controllable, local-first personal agent that remembers you — and owns its own
execution loop instead of wrapping a framework.

Alpha runs from your terminal, keeps everything in a local SQLite database, and
builds up a durable memory of your conversations in the background. Point it at a
mock model to try it in seconds with no API key, then switch to a real provider
when you're ready.

## Why Alpha Agent

- **It remembers.** Alpha learns facts, preferences, and constraints from your
  conversations and stores them as durable memory. A background service quietly
  extracts, consolidates, and de-conflicts that memory while you work — no manual
  curation required.
- **Local-first and private.** All state lives in one SQLite file under
  `~/.alpha-agent/`. Nothing leaves your machine except the LLM calls you choose
  to make.
- **No framework lock-in.** Alpha owns its turn execution and tool loop directly
  — it is *not* a LangChain / LangGraph / LlamaIndex / AutoGen / CrewAI wrapper.
  Every turn is explicit, bounded, and auditable.
- **Bring your own model.** Swap between a built-in `mock`, any
  OpenAI-compatible API, DeepSeek, or Codex with a single config change.
- **Real tools, safely gated.** Built-in web search (via Tavily) and an opt-in
  local `bash` tool, plus an explicit memory recall/propose path the model calls
  on demand.
- **Always-on daemon.** A background runtime owns sessions and keeps doing memory
  work between your messages.

## Quickstart

Try it end-to-end with the built-in mock model — no API key needed:

```bash
uv sync                      # install dependencies
uv run alpha init            # create the local SQLite database + config
uv run alpha daemon start    # start the background runtime
uv run alpha ask "hello"     # send a single message
```

Or open an interactive session:

```bash
uv run alpha chat
```

That's it — you're talking to the agent. The mock provider gives canned replies
so you can verify everything is wired up before adding a real model.

## Talk to a real model

Pick a provider, give it credentials, and restart the daemon so it picks up the
new settings:

```bash
# DeepSeek
uv run alpha config set llm.provider deepseek
uv run alpha config set deepseek.api_key sk-...
uv run alpha daemon restart

# Any OpenAI-compatible API
uv run alpha config set llm.provider openai-compatible
uv run alpha config set compatible.base_url https://api.openai.com/v1
uv run alpha config set compatible.api_key sk-...
uv run alpha daemon restart

# Codex (OAuth) — easiest if you've logged in with the Codex CLI
uv run alpha config set llm.provider codex
uv run alpha daemon restart
```

Then `uv run alpha ask "..."` or `uv run alpha chat` as usual.

> The daemon reads provider settings at startup. **After any provider or
> credential change, run `alpha daemon restart`.**

## Everyday commands

```bash
# Conversation
uv run alpha ask "what did we decide yesterday?"
uv run alpha chat --session <session-id>   # resume a past session

# Daemon lifecycle
uv run alpha daemon start | status | restart | stop
uv run alpha daemon run                     # foreground (for supervisors/debug)

# Config
uv run alpha config show
uv run alpha config set llm.provider codex
uv run alpha config get llm.provider

# Inspect what the agent knows / does
uv run alpha skills list
uv run alpha debug prompt "summarize this session" --session <id> --trace
uv run alpha cognition consolidate --now --dry-run
```

Run `uv run alpha --help` (or `--help` on any subcommand) for the full list.

## Built-in tools

Memory recall, memory proposal, and local file inspection are available to the
model by default. Other tools are opt-in:

- **Web search and fetch** — enabled automatically once a Tavily key is set:

  ```bash
  uv run alpha config set tavily.api_key tvly-...
  uv run alpha daemon restart
  ```

- **Local `bash`** — disabled by default. Enable it only for trusted local use
  when you want the agent to run build, test, or diagnostic commands:

  ```bash
  uv run alpha config set tools.bash.enabled true
  uv run alpha daemon restart
  ```

  The tool runs with a cleaned environment, timeouts, dangerous-command
  blocking, and output truncation — but it is **not a security sandbox**. Don't
  expose it to untrusted gateway users without a stronger approval layer.

- **Local file tools** — enabled by default for the Alpha workspace under
  `runtime.home_dir`. Relative local paths in runtime, bash, and file-tool
  settings resolve under that home directory, so daemon behavior does not depend
  on the directory where it was started. Configure the allowed roots, or disable
  file tools if you do not want the model to inspect local files:

  ```bash
  uv run alpha config set runtime.home_dir ~/.alpha-agent
  uv run alpha config set tools.files.allowed_roots workspace
  uv run alpha config set tools.files.enabled false
  uv run alpha daemon restart
  ```

  Read-only file inspection tools reject paths outside `tools.files.allowed_roots`,
  skip common large internal directories, reject binary content, and apply
  configured output limits such as `tools.files.max_glob_results`,
  `tools.files.max_search_results`, `tools.files.max_read_lines`, and
  `tools.files.max_output_chars`.

  `file_patch` is a separate write tool and is disabled by default. It is only
  registered when `tools.files.enabled = true`, `tools.files.patch_enabled = true`,
  and `tools.files.write_roots` is non-empty:

  ```bash
  uv run alpha config set tools.files.write_roots workspace
  uv run alpha config set tools.files.patch_enabled true
  uv run alpha daemon restart
  ```

  `file_patch` validates paths against `tools.files.write_roots`, rejects symlink
  targets, binary files, and files above `tools.files.max_file_bytes`, and requires
  `expected_sha256` to match existing file content before edits are applied. New
  files require `create_if_missing = true` and an empty or omitted
  `expected_sha256`. Whole-file creation can create missing parent directories
  only when `tools.files.create_parent_dirs_enabled` is enabled.

Each registered tool declares one `ToolSpec`: provider-facing name,
description, parameters, and strict mode plus internal governance fields such as
toolset, read/write behavior, concurrency safety, destructive side effects, and
maximum model-visible result size. Availability stays dynamic via
`check_available()`. Runtime traces record the spec and availability; provider
tool schemas are projected only from name, description, parameters, and strict
mode. Tool specs do not use a `group` field.

## How it works

- **Daemon-owned turns.** `alpha daemon start` runs the single process that owns
  sessions. `ask` and `chat` are thin clients that talk to it over a local
  socket. Each turn runs a bounded LLM + tool loop and persists every message.
- **Memory that builds itself.** Turns are appended to a local event log. A
  background cognition service periodically intakes new conversation, extracts
  durable memories (facts, preferences, constraints, procedures, values,
  relationships), consolidates them, and reviews conflicts.
- **Explicit recall.** The model pulls relevant memory on demand via a
  `memory_recall` tool and writes updates via `memory_propose` — recall is never
  silently injected. A compact, stable profile of who you are is always kept near
  the top of the prompt.
- **Pluggable providers.** `mock`, `openai-compatible`, `deepseek`, and `codex`
  share one interface; the rest of the runtime doesn't care which you use.

## Configuration

Alpha reads long-lived settings from `~/.alpha-agent/config.toml`. Manage it with
the CLI:

```bash
uv run alpha config init     # create the file
uv run alpha config show     # print effective settings (secrets masked)
uv run alpha config set <section.key> <value>
uv run alpha config get <section.key>
```

Environment variables and `.env` override the file for one-off runs and secrets.
Precedence is:

```text
defaults < config.toml < .env / environment variables
```

See [`config.example.toml`](config.example.toml) for every available key with
inline comments, and [`.env.example`](.env.example) for environment variables
read by the loader. Common starting points: `llm.provider`, `llm.model`,
`tools.bash.enabled`, and `tavily.api_key`.

## Status & roadmap

Alpha is an evolving baseline, intentionally kept small and controllable. Today:

- Background memory extraction, consolidation, and conflict review run
  automatically; production budget/rate controls are still to come.
- The Drive Loop (autonomous goal pursuit) is synchronous and **disabled by
  default** — one self-signal per manual `cognition drive --once` pass.
- No web UI, no multi-agent system, and no real Feishu/WeChat adapter yet (the
  gateway shell and diagnostics exist; platform adapters do not).

Planned next: local file/note ingestion, an API server, a web UI, and channel
integrations.

## Development

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run mypy src tests      # type-check
```
