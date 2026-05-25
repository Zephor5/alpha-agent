## Rules for this project
- DO NOT CONSIDER COMPATIBILITY WHEN CODING, INCLUDING EXISTED DATA IN DATABASE
- DO NOT FOLLOW A "PARTIAL COMPATIBILITY FIRST, GRADUAL REPLACEMENT LATER" APPROACH; PRIORITIZE DIRECT REFACTORING TOWARD THE TARGET ARCHITECTURE.
- ANY MODIFICATIONS MUST BE CONSIDERED FROM A GLOBAL PERSPECTIVE, TAKING INTO ACCOUNT THE ENTIRE PROJECT, ALL MODULES, AND THE ASSOCIATED IMPACTS ON DOCUMENTATION.
- WHEN THE SAME OR HIGHLY SIMILAR LOGIC APPEARS 3 OR MORE TIMES, EXTRACT IT PROMPTLY INTO A SHARED FUNCTION, MODULE, OR MECHANISM INSTEAD OF KEEPING SIMILAR REUSED CODE IN MULTIPLE PLACES.
- DO NOT INCLUDE LOCAL MACHINE-SPECIFIC ABSOLUTE PATHS IN THE REPOSITORY. USE RELATIVE PATHS, PROJECT-ROOT-RELATIVE PATHS, ENVIRONMENT VARIABLES, OR GENERIC USER-HOME BASED PATHS INSTEAD.

## Project Content Navigation
```text
README.md             Project overview, install steps, CLI usage, configuration, state baseline, and current limitations.
pyproject.toml        Package metadata, Python version, dependencies, console script entry point, and lint/type-check settings.
config.example.toml   Example runtime configuration.
docs/
  cognition/          Reference docs for cognition related model.
  develop_record/     Archived working notes and completed refactor records, no need to read this unless required.
  doing/              Execution ledger for active tasks only; record during execution, then clear after self-check on completion
  todo/               Todo docs
    cognition-runtime/ Staged plan for rebuilding long-term cognition after Phase 00 cleanup.
src/
  alpha_agent/
    cli.py            Typer CLI entry point for chat, ask, config, skills, debug, and gateway commands.
    config.py         Runtime configuration loading, defaults, environment overrides, and persistent config handling.
    cognition/        Cognition foundations, Reactive loop wiring, belief/context/reflection projections, and L1 reflectors.
      controller.py   CognitiveController orchestration for Reactive single tick.
      stages/         Perceive, Attend, Interpret, Judge, Decide, Act, Feedback, Reflect, and Revise stages.
      reflectors/     Deterministic L1 reflection rules for read-only Reactive tick audits.
      loops/          In-process Scheduler, checkpoint storage, ConsolidationLoop, and deterministic consolidation workers.
      projections/    Subject/counterpart projections, SQLite-backed BeliefProjection, ProcedureProjection, ReflectionProjection, and ContextWindowProjection.
      render/         CognitionView assembly and renderers for chat prompts, graph snapshots, diffs, and evidence traces.
    runtime/          Core turn/session execution, event models, session context, and runtime tool wiring.
      counterpart_router.py Source metadata to CounterpartRef routing and first-observed event handling.
    state/            SQLite-backed runtime tables plus cognition event/projection tables, including cognitive_events, counterpart_view, belief_view, context_window_view, context_window_background, procedure_view, cognition_worker_checkpoint, and reflection_view.
    llm/              LLM provider interface and concrete providers, including mock, OpenAI-compatible, DeepSeek, and Codex.
    gateway/          Gateway operation shell, session routing, adapter contracts, status, logging, and gateway config.
    tools/            Tool abstractions and registry used by the runtime.
    skills/           Procedural skill manager and built-in Markdown skills.
    utils/            Shared utility helpers for IDs, text, and time.
tests/                Test coverage grouped around CLI, runtime loop, cognition renderers, state, LLM providers, config, gateway, and future cognition.
```
