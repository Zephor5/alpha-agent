## Rules for this project
- DO NOT CONSIDER COMPATIBILITY WHEN CODING, INCLUDING EXISTED DATA IN DATABASE
- DO NOT FOLLOW A "PARTIAL COMPATIBILITY FIRST, GRADUAL REPLACEMENT LATER" APPROACH; PRIORITIZE DIRECT REFACTORING TOWARD THE TARGET ARCHITECTURE.
- ANY MODIFICATIONS MUST BE CONSIDERED FROM A GLOBAL PERSPECTIVE, TAKING INTO ACCOUNT THE ENTIRE PROJECT, ALL MODULES, AND THE ASSOCIATED IMPACTS ON DOCUMENTATION.
- WHEN THE SAME OR HIGHLY SIMILAR LOGIC APPEARS 3 OR MORE TIMES, EXTRACT IT PROMPTLY INTO A SHARED FUNCTION, MODULE, OR MECHANISM INSTEAD OF KEEPING SIMILAR REUSED CODE IN MULTIPLE PLACES.
- DO NOT INCLUDE LOCAL MACHINE-SPECIFIC ABSOLUTE PATHS IN THE REPOSITORY. USE RELATIVE PATHS, PROJECT-ROOT-RELATIVE PATHS, ENVIRONMENT VARIABLES, OR GENERIC USER-HOME BASED PATHS INSTEAD.

## Project Content Navigation
- `README.md`: Project overview, install steps, CLI usage, configuration, retrieval behavior, and current limitations.
- `pyproject.toml`: Package metadata, Python version, dependencies, console script entry point, and lint/type-check settings.
- `config.example.toml`: Example runtime configuration.
- `docs/memory_design.md`: Memory architecture and design notes.
- `docs/TODO.md`: Current roadmap and integration-oriented follow-up work.
- `src/alpha_agent/cli.py`: Typer CLI entry point for chat, ask, memory, config, skills, debug, and gateway commands.
- `src/alpha_agent/config.py`: Runtime configuration loading, defaults, environment overrides, and persistent config handling.
- `src/alpha_agent/runtime/`: Core turn/session execution, event models, prompt building, and runtime tool wiring.
- `src/alpha_agent/memory/`: SQLite-backed memory system, schema, persistence, retrieval, salience, extraction, consolidation, and memory-layer implementations.
- `src/alpha_agent/llm/`: LLM provider interface and concrete providers, including mock, OpenAI-compatible, DeepSeek, and Codex.
- `src/alpha_agent/gateway/`: Gateway operation shell, session routing, adapter contracts, status, logging, and gateway config.
- `src/alpha_agent/tools/`: Tool abstractions and registry used by the runtime.
- `src/alpha_agent/skills/`: Procedural skill manager and built-in Markdown skills.
- `src/alpha_agent/graph/`: Lightweight graph models and storage utilities.
- `src/alpha_agent/utils/`: Shared utility helpers for IDs, text, and time.
- `tests/`: Test coverage grouped around CLI, runtime loop, prompt building, memory, retrieval, LLM providers, config, gateway, and consolidation.
