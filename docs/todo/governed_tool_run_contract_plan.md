# GovernedTool Run Contract Implementation Plan

## Objective

Make the tool execution contract explicit and executable.

Concrete tools should not implement `run()` directly. They should implement lifecycle
hooks. The base class owns execution order, cancellation checkpoints, expected failure
mapping, JSON-safe payload validation, and `ToolResult` construction.

## Target Contract

Add a `GovernedTool` base class for tools that need enforced run semantics.

Required behavior:

- `run()` is final and implemented by `GovernedTool`.
- Concrete tools parse arguments into a typed request object.
- Safety and policy checks run before external calls or local side effects.
- External dependency failures become provider-neutral structured outputs.
- Invalid model arguments remain failures, not structured business outputs.
- Unexpected implementation errors escape to the executor.
- `ToolResult.name` is always `self.spec.name`.
- `ToolResult.output` and `ToolResult.metadata` are JSON-safe.
- Cancellation is checked before parsing and before execution.

## Types

Add these to `src/alpha_agent/tools/base.py`, or to
`src/alpha_agent/tools/governed.py` with re-exports from `base.py` if the base module
becomes too large.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, final

RequestT = TypeVar("RequestT")


@dataclass(frozen=True)
class ToolPayload:
    output: JSONValue
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


class ToolInputError(ValueError):
    """Raised when model-supplied tool arguments are invalid."""


class ToolBlockedError(Exception):
    """Raised when a policy or safety preflight blocks execution."""

    public_message: str


class ExternalToolError(Exception):
    """Raised when an external dependency fails in an expected way."""

    public_message: str
    error_type: str
    http_status: int | None


class ToolInvariantError(RuntimeError):
    """Raised when a tool implementation violates the base contract."""
```

## GovernedTool API

```python
class GovernedTool(Generic[RequestT], ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        ...

    @abstractmethod
    def check_available(self) -> ToolAvailability:
        ...

    @final
    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        ...

    @abstractmethod
    def parse_arguments(self, arguments: Mapping[str, Any]) -> RequestT:
        ...

    def check_runtime_ready(self, request: RequestT, context: ToolExecutionContext) -> None:
        ...

    def preflight(self, request: RequestT, context: ToolExecutionContext) -> None:
        ...

    @abstractmethod
    def execute(self, request: RequestT, context: ToolExecutionContext) -> ToolPayload:
        ...

    def external_failure(
        self,
        request: RequestT,
        error: ExternalToolError,
        context: ToolExecutionContext,
    ) -> ToolPayload:
        ...

    def blocked_failure(
        self,
        request: RequestT,
        error: ToolBlockedError,
        context: ToolExecutionContext,
    ) -> ToolPayload:
        ...
```

## Run Order

`GovernedTool.run()` must execute in this order:

1. `context.check_canceled("before_tool_parse")`
2. `request = parse_arguments(arguments)`
3. `check_runtime_ready(request, context)`
4. `preflight(request, context)`
5. `context.check_canceled("before_tool_execute")`
6. `payload = execute(request, context)`
7. Convert expected failures:
   - `ExternalToolError` -> `external_failure(...)`
   - `ToolBlockedError` -> `blocked_failure(...)`
8. Validate `ToolPayload.output` and `ToolPayload.metadata` as JSON-safe.
9. Return `ToolResult(name=self.spec.name, output=payload.output, metadata=payload.metadata)`.

## Failure Semantics

Invalid model arguments:

- Raised as `ToolInputError` or `ValueError`.
- Allowed to escape to `ToolExecutor`.
- Traced as `tool.failed`.

Policy or safety blocks:

- Raised as `ToolBlockedError`.
- Converted through `blocked_failure(...)` when the tool has a useful structured blocked
  output.

External dependency failures:

- Raised as `ExternalToolError`.
- Converted through `external_failure(...)`.
- Model-visible output must not expose provider names, backend endpoints, secrets, or
  implementation-specific transport details.

Implementation errors:

- Raised as ordinary exceptions or `ToolInvariantError`.
- Allowed to escape.
- Must not be disguised as user-correctable tool output.

## JSON Safety

Add:

```python
def ensure_json_value(value: Any, *, field_name: str = "output") -> JSONValue:
    ...
```

Rules:

- Accept `None`, `bool`, `int`, `float`, `str`, lists, and dicts with string keys.
- Recursively validate list items and dict values.
- Reject arbitrary objects, `Path`, `bytes`, exceptions, and mappings with non-string keys.
- Raise `ToolInvariantError` with a concise message.

Use this helper inside `GovernedTool.run()` before constructing `ToolResult`.

## Migration Plan

### Task 1: Add GovernedTool Foundation

Files:

- `src/alpha_agent/tools/base.py`
- `tests/test_tool_governance.py`

Implementation:

- Add `ToolPayload`.
- Add `ToolInputError`, `ToolBlockedError`, `ExternalToolError`, and
  `ToolInvariantError`.
- Add `ensure_json_value(...)`.
- Add `GovernedTool` with final `run()`.
- Keep the existing `Tool` protocol and registry behavior unchanged.

Acceptance criteria:

- Existing tools still register and execute without migration.
- `GovernedTool.run()` constructs `ToolResult` with `spec.name`.
- Invalid payload output or metadata raises `ToolInvariantError`.
- `ToolInputError` and `ValueError` escape.
- `ExternalToolError` requires an `external_failure(...)` mapper.
- Cancellation is checked before parse and before execute.

Verification:

```bash
uv run pytest tests/test_tool_governance.py -q
```

### Task 2: Migrate web_fetch

Files:

- `src/alpha_agent/tools/web_fetch.py`
- `tests/test_web_fetch_tool.py`

Implementation:

- Make `TavilyWebFetchTool` inherit `GovernedTool[_FetchRequest]`.
- Remove its direct `run()` implementation.
- Move request parsing into `parse_arguments(...)`.
- Move API-key readiness into `check_runtime_ready(...)`.
- Move URL safety into `preflight(...)`.
- Move provider call and response normalization into `execute(...)`.
- Convert provider HTTP, timeout, request, invalid JSON, and invalid response failures into
  `ExternalToolError`.
- Implement `external_failure(...)` using the existing provider-neutral failed output
  shape.

Acceptance criteria:

- Provider-facing schema remains unchanged.
- Success output remains unchanged.
- Provider failure output remains structured and provider-neutral.
- URL safety still runs before any provider request.
- Metadata may keep provider diagnostics.

Verification:

```bash
uv run pytest tests/test_web_fetch_tool.py tests/test_url_safety.py -q
```

### Task 3: Migrate web_search

Files:

- `src/alpha_agent/tools/web_search.py`
- `tests/test_web_search_tool.py`

Implementation:

- Add a request dataclass for normalized search arguments.
- Move argument normalization into `parse_arguments(...)`.
- Move API-key readiness into `check_runtime_ready(...)`.
- Convert provider HTTP, timeout, request, invalid JSON, and invalid response failures into
  `ExternalToolError`.
- Implement a provider-neutral failed output shape:

```json
{
  "answer": null,
  "query": "...",
  "request_id": null,
  "response_time": null,
  "results": [],
  "status": "failed",
  "error": "Web search failed with HTTP status 500."
}
```

Acceptance criteria:

- ToolSpec remains provider-neutral.
- Current success output remains unchanged.
- External failures do not expose provider endpoints.

Verification:

```bash
uv run pytest tests/test_web_search_tool.py -q
```

### Task 4: Migrate bash

Files:

- `src/alpha_agent/tools/bash.py`
- `tests/test_bash_tool.py`

Implementation:

- Add a request dataclass containing the policy-prepared shell request and secret values.
- Move `BashExecutionPolicy.prepare(...)` into `preflight(...)`.
- Convert `BashPolicyError` into `ToolBlockedError`.
- Move backend execution and output governance into `execute(...)`.
- Implement `blocked_failure(...)` using the current blocked-result output shape.
- Keep `trace_arguments(...)` unchanged.

Acceptance criteria:

- Blocked command output remains unchanged.
- Command execution output remains unchanged.
- Redaction and truncation remain unchanged.

Verification:

```bash
uv run pytest tests/test_bash_tool.py -q
```

### Task 5: Migrate File Tools

Files:

- `src/alpha_agent/tools/files/tools.py`
- `tests/test_file_tools.py`

Implementation:

- Migrate one file tool at a time, starting with `FileGlobTool`.
- Preserve each tool's output shape.
- Use `ToolInputError` for invalid argument shape.
- Use `ToolBlockedError` for path policy, symlink, binary, or size blocks when the tool
  should return structured blocked output.
- Keep local path display behavior relative to configured roots.

Acceptance criteria:

- Provider-facing schemas remain unchanged.
- Existing output shapes remain unchanged.
- Outputs do not introduce local machine-specific absolute paths.

Verification:

```bash
uv run pytest tests/test_file_tools.py -q
```

### Task 6: Migrate Memory Tools

Files:

- `src/alpha_agent/tools/memory_recall.py`
- `src/alpha_agent/tools/memory_propose.py`
- Memory-related tests under `tests/`.

Implementation:

- Migrate after web, bash, and file tools pass.
- Preserve output shapes.
- Keep memory LLM-directed; do not add hidden recall or hidden writes.

Acceptance criteria:

- Existing memory behavior remains unchanged.
- Cognition memory tests pass.

Verification:

```bash
uv run pytest tests/cognition/test_memory_recall_tool.py -q
rg "MemoryPropose|memory_propose" tests
```

Run the relevant memory proposal tests found by `rg`.

## Tool Author Rules

After migration, tools inheriting `GovernedTool` must follow these rules:

- Do not override `run()`.
- `parse_arguments(...)` returns a typed request object.
- `preflight(...)` performs safety and policy checks before side effects.
- `execute(...)` returns `ToolPayload`.
- External dependency failures become `ExternalToolError`.
- External-provider tools implement `external_failure(...)`.
- Tool output is provider-neutral unless the public tool contract explicitly says
  otherwise.
- Metadata may include non-secret diagnostic details.
- Output and metadata must not include secrets or local machine-specific absolute paths.

## Full Verification Gate

Run after each migrated tool family:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## Completion Criteria

- All registered built-in tools either inherit `GovernedTool` or have a documented reason
  to remain on the raw `Tool` protocol.
- No migrated tool overrides `run()`.
- External-provider tools have provider-neutral failure outputs.
- Policy and safety preflight behavior is covered by tests.
- JSON payload validation is covered by tests.
- Full CI gate passes.
