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
  cognition/          Reference docs for cognition and memory design.
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
      counterpart_router.py Source metadata to CounterpartRef routing and first-observed event handling.
    cognition/        Cognition foundations, Reactive tick orchestration, event emission, loop coordination, payload contracts, projections, and search/tokenization helpers.
      controller.py   CognitiveController orchestration for one Reactive tick.
      coordinator.py  LoopCoordinator lock/lease control for Reactive and background cognition loops.
      counterpart_profile.py Counterpart digest belief helpers.
      emitter.py      Cognitive event emission helpers.
      payload_contract.py Fail-fast validation for consumed cognition event payload fields.
      projection_runner.py Projection registry execution and rebuild helpers.
      search_tokenizer.py Deterministic tokenization for mixed CJK and technical search text.
      models/         Frozen cognition data contracts for events, perceptions, judgments, decisions, beliefs, goals, strategies, subjects, values, situations, and threads.
      event_log/      In-memory and SQLite cognitive event log implementations.
      stages/         Perceive, Attend, Interpret, Judge, Decide, Act/Effector, Feedback, Reflect, and Revise stages.
      reflectors/     L1/L2/L3 reflector orchestration, deterministic audit rules, L2 strategy rules, and L3 self-model aggregators.
        l2_rules/     Deterministic L2 strategy override rules.
        l3_aggregators/ Deterministic self-model aggregators for capabilities, failure modes, preferences, strategies, tradeoffs, and interaction patterns.
      loops/          In-process scheduler, checkpoint storage, ConsolidationLoop, deterministic workers, and synchronous DriveLoop.
        workers/      Background consolidation workers for beliefs, context, procedures, goals, value lens, strategies, counterpart summaries, and archive/resolve tasks.
      goals/          GoalRegistry event write path for DriveLoop goals.
      value/          Deterministic ValueProfile derivation, ValueLens persistence, and conflict resolution.
      projections/    SQLite-backed projections for counterpart, belief, context window, reflection, strategy, goal, subject, procedure, and event counts.
      render/         CognitionView assembly and renderers for chat prompts, graph snapshots, diffs, and evidence traces.
    state/            SQLite-backed state store/schema/models for session messages, runtime traces, gateway mappings/dedup, cognitive events, and projection tables.
    llm/              LLM provider interface and concrete providers, including mock, OpenAI-compatible, DeepSeek, and Codex.
    tools/            Tool abstractions and registry used by the runtime.
      shell/          Structured local shell execution backend, output capture, policy, and command semantics.
    skills/           Procedural skill manager and built-in Markdown skills.
      builtin/        Built-in Markdown skills such as debug-loop and summarize.
    utils/            Shared utility helpers for IDs and time.
tests/                Test coverage for CLI, runtime, config, daemon, gateway, LLM providers, tools, and session/context behavior.
  cognition/          Cognition-specific tests for events, projections, renderers, reflectors, loops, goals, memory tools, tokenization, and CLI inspection commands.
```

## Validation Commands
Run these from the project root to mirror the current CI gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```
