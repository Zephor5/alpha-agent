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
.github/workflows/    GitHub Actions CI workflow definition.
.gitignore            Ignore rules for local environments, caches, build artifacts, and runtime state.
LICENSE               Project license.
docs/
  cognition/          Ideal cognition descriptions, just for the big picture of the design.
  develop_record/     Historical archive only; skip by default and never treat as current implementation evidence.
  doing/              Execution ledger for active tasks only; record during execution, then clear after self-check.
  todo/               Planning docs.
src/
  alpha_agent/
    cli.py            Typer CLI for init, ask/chat, config, daemon, gateway, skills, debug, cognition import/inspection, and Drive goals.
    config.py         Runtime configuration loading, defaults, environment overrides, and persistent config handling.
    conversation_import/ DeepSeek export conversion into normalized session/import records.
    daemon/           Local daemon process lifecycle, IPC client/server, runtime loop, status, manager, and import service.
    gateway/          Gateway operation shell, adapter contracts, session routing, status, logging, and gateway runtime config.
      adapters/       External gateway adapter interfaces.
    runtime/          Agent turn/session execution, prompts, events, context budget/handover, session context, counterpart routing, and tool wiring.
      chat_messages.py ChatMessage formatting, source replay conversion, system-reminder stripping, and token estimates.
      counterpart_router.py Source metadata to CounterpartRef routing and first-observed event handling.
      prompt_builder.py Answer prompt assembly from session context, memory, and runtime state.
    cognition/        Cognition event-sourced memory, reactive/background loop coordination, projections, and Drive goals.
      __init__.py     Cognition package export surface.
      authority.py    Authority and consent helpers for cognition writes.
      background_llm_contract.py JSON schemas and validators for LLM-mediated extraction, consolidation, summaries, and feedback attribution.
      controller.py   Default cognition projection registry construction.
      coordinator.py  LoopCoordinator lock, priority, yield, and lease control for cognition loops.
      domain_guidance.py Active summary-derived guidance and memory proposal confirmation policy.
      emitter.py      Cognitive event emission helpers.
      payload_contract.py Fail-fast validation for consumed cognition event payload fields.
      processing_ledger.py Background source/window/stage ledger, status tracking, idempotency keys, and recovery helpers.
      projection_runner.py Projection registry execution and projection rebuild helpers.
      search_tokenizer.py Deterministic tokenization for mixed CJK and technical search text.
      source_time.py  Source message and belief time-range resolution for prompts and audit text.
      state_service.py Canonical cognition state write/query facade for atomic beliefs, summaries, background operations, feedback, and audits.
      event_log/
        __init__.py   Event log package exports.
        base.py       EventLog protocol.
        memory.py     In-memory event log for focused tests and ephemeral runs.
        sqlite.py     SQLite-backed cognitive event log persisted through StateStore.
      goals/
        __init__.py   Goal registry package exports.
        registry.py   GoalRegistry event write path for Drive Loop goal lifecycle and progress events.
      loops/
        __init__.py   Loop service, Drive Loop, feedback attribution, and scheduler exports.
        README.md     Notes on retained loop infrastructure and worker direction.
        background_service.py Daemon-owned background cognition drain for extraction, consolidation, conflict review, summaries, and archival.
        compact_extraction.py Direct extraction service for compacted session outputs.
        drive.py      Synchronous Drive Loop that turns eligible active goals into self-signal turns.
        feedback_attribution.py Realtime attribution of user feedback to recalled beliefs from the previous turn.
        scheduler.py  Worker checkpoint storage, worker reports, and scheduled worker protocols.
        workers/
          __init__.py Worker exports and default worker list.
          _common.py  Shared worker report, cursor, prompt JSON, and trace metadata helpers.
          archive_expired.py Worker that archives expired beliefs.
          memory_extraction.py LLM-mediated source-window extraction of atomic belief drafts.
          memory_consolidation.py LLM-mediated consolidation and conflict review for active beliefs.
          memory_summary.py LLM-mediated summary belief generation and refresh.
      models/
        __init__.py   Public cognition model export surface.
        _ids.py       Typed reference/id helpers for subjects, counterparts, beliefs, entities, situations, and actors.
        _serialization.py Dataclass record serialization helpers.
        belief.py     AtomicBelief, SummaryBelief, validity windows, relation records, and belief validation.
        counterpart.py Counterpart relationship, service commitment, and style hint data contracts.
        enums.py      Cognition enums for memory kinds, authority, lifecycle, event kinds, loop priority, and stimulus kinds.
        event.py      CognitiveEvent event-sourcing contract.
        goal.py       Drive Loop goal contract.
        perception.py Perception data contract.
        situation.py  Situation, social context, and authority hint contracts.
        subject.py    Subject data contract and self-subject constant.
      projections/
        __init__.py   Projection package export surface.
        base.py       Projection and EventProjection base protocols.
        registry.py   Projection registration and typed lookup.
        belief.py     SQLite-backed belief projection, FTS search, recall ranking inputs, and rebuild behavior.
        counterpart.py Counterpart projection from observed counterpart events.
        event_count.py Event count projection by cognitive event kind.
        goal.py       Goal lifecycle projection and active-goal queries.
        subject.py    Subject projection from subject/situation events.
    state/            SQLite-backed state store/schema/models for session messages, runtime traces, gateway mappings/dedup, cognitive events, and projection tables.
    llm/              LLM provider interface, chat-completion adapters, tracing, and concrete mock, OpenAI-compatible, DeepSeek, MiMo, and Codex providers.
    tools/            Tool abstractions, default registry, bash/web tools, URL safety, memory recall/propose tools, and file tools.
      files/          Sandboxed file glob/read/search/patch/write tools plus path validation, atomic IO, and patch planning.
      shell/          Structured local shell execution backend, output capture, policy, and command semantics.
    skills/           Procedural skill manager and built-in Markdown skills.
      builtin/        Built-in Markdown skills such as debug-loop and summarize.
    utils/            Shared utility helpers for IDs and time.
tests/                Test coverage for CLI, runtime, config, daemon, gateway, LLM providers, tools, imports, and session/context behavior.
  cognition/          Cognition-specific tests for event logs, payload contracts, projections, loop coordination, background workers, goals, memory tools, feedback attribution, source time, and tokenization.
```

## Validation Commands
Run these from the project root to mirror the current CI gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```
