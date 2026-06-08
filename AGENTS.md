## Rules for this project
- DO NOT CONSIDER COMPATIBILITY WHEN CODING, INCLUDING EXISTED DATA IN DATABASE
- DO NOT FOLLOW A "PARTIAL COMPATIBILITY FIRST, GRADUAL REPLACEMENT LATER" APPROACH; PRIORITIZE DIRECT REFACTORING TOWARD THE TARGET ARCHITECTURE.
- ANY MODIFICATIONS MUST BE CONSIDERED FROM A GLOBAL PERSPECTIVE, TAKING INTO ACCOUNT THE ENTIRE PROJECT, ALL MODULES, AND THE ASSOCIATED IMPACTS ON DOCUMENTATION.
- WHEN THE SAME OR HIGHLY SIMILAR LOGIC APPEARS 3 OR MORE TIMES, EXTRACT IT PROMPTLY INTO A SHARED FUNCTION, MODULE, OR MECHANISM INSTEAD OF KEEPING SIMILAR REUSED CODE IN MULTIPLE PLACES.
- DO NOT INCLUDE LOCAL MACHINE-SPECIFIC ABSOLUTE PATHS IN THE REPOSITORY. USE RELATIVE PATHS, PROJECT-ROOT-RELATIVE PATHS, ENVIRONMENT VARIABLES, OR GENERIC USER-HOME BASED PATHS INSTEAD.
- Treat `docs/develop_record/` as historical archive only. Do not read it during normal implementation, review, or current-state analysis unless the user explicitly asks for historical context or the task truly requires archaeology. Never use `docs/develop_record/` as evidence of current behavior, current architecture, or live requirements without verifying against active code and non-archived docs.

## Project Content Navigation
```text
AGENTS.md             Project-specific agent rules and content map.
README.md             Project overview, install steps, CLI usage, configuration, state baseline, and current limitations.
pyproject.toml        Package metadata, Python version, dependencies, console script entry point, and lint/type-check settings.
uv.lock               Locked dependency graph for uv-based installs.
config.example.toml   Example runtime configuration.
.env.example          Environment variable template for local runtime paths, LLM providers, and memory limits.
.github/              GitHub Actions CI workflow definition.
.gitignore            Ignore rules for local environments, caches, build artifacts, and runtime state.
LICENSE               Project license.
docs/
  cognition/          Reference docs for cognition.
  develop_record/     Historical archive only; skip by default and never treat as current implementation evidence.
  doing/              Execution ledger for active tasks only; record during execution, then clear after self-check.
  todo/               Project todo docs.
src/
  alpha_agent/
    cli.py            Typer CLI entry point for init, ask/chat, config, daemon, gateway, skills, debug, cognition, goals, lens, and self-model commands.
    config.py         Runtime configuration loading, defaults, environment overrides, and persistent config handling.
    daemon/           Local daemon process lifecycle, IPC client/server, runtime loop, status, and manager.
    gateway/          Gateway operation shell, adapter contracts, session routing, status, logging, and gateway config.
      adapters/       External gateway adapter interfaces.
    runtime/          Agent turn/session execution, event models, context budget/handover, session context, counterpart routing, and runtime tool wiring.
      chat_messages.py ChatMessage formatting, source replay conversion, system-reminder helpers, and chat token estimates.
      counterpart_router.py Source metadata to CounterpartRef routing and first-observed event handling.
    cognition/        Cognition foundations, Reactive/background loop orchestration, event emission, state services, payload contracts, projections, and search/tokenization helpers.
      authority.py    Authority and consent helpers for cognition writes.
      background_llm_contract.py Validation contracts for LLM-mediated background cognition outputs.
      controller.py   CognitiveController orchestration for one Reactive tick.
      coordinator.py  LoopCoordinator lock/lease control for Reactive and background cognition loops.
      domain_guidance.py Domain guidance assembly for background cognition prompts.
      emitter.py      Cognitive event emission helpers.
      payload_contract.py Fail-fast validation for consumed cognition event payload fields.
      processing_ledger.py Background processing ledger stage/status helpers.
      projection_runner.py Projection registry execution and rebuild helpers.
      search_tokenizer.py Deterministic tokenization for mixed CJK and technical search text.
      state_service.py State service for atomic beliefs, summaries, background operations, and audit writes.
      models/         Frozen cognition data contracts for events, beliefs, goals, subjects, situations, enums, and loop metadata.
      event_log/      In-memory and SQLite cognitive event log implementations.
      loops/          In-process scheduler, checkpoint storage, ConsolidationLoop, background service, and DriveLoop.
        workers/      LLM-mediated memory extraction, consolidation, summary workers, plus expired-belief archival.
      goals/          GoalRegistry event write path for DriveLoop goals.
      projections/    SQLite-backed projections for counterpart, belief, goal, subject, and event counts.
    state/            SQLite-backed state store/schema/models for session messages, runtime traces, gateway mappings/dedup, cognitive events, and projection tables.
    llm/              LLM provider interface and concrete providers, including mock, OpenAI-compatible, DeepSeek, and Codex.
    tools/            Tool abstractions and registry used by the runtime.
      shell/          Structured local shell execution backend, output capture, policy, and command semantics.
    skills/           Procedural skill manager and built-in Markdown skills.
      builtin/        Built-in Markdown skills such as debug-loop and summarize.
    utils/            Shared utility helpers for IDs and time.
tests/                Test coverage for CLI, runtime, config, daemon, gateway, LLM providers, tools, and session/context behavior.
  cognition/          Cognition-specific tests for events, projections, renderers, loops, goals, memory tools, tokenization, and CLI inspection commands.
```

## Validation Commands
Run these from the project root to mirror the current CI gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```
