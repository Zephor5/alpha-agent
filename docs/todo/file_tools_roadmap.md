# File Tools Roadmap

## Purpose

Build Alpha Agent's file tools into a safe, high-signal coding interface instead
of a thin filesystem wrapper.

The target is not to copy Hermes or Claude Code tool-for-tool. The target is a
coherent Alpha toolset with:

- Fast discovery before reading large files.
- Bounded, paginated, line-addressable reads.
- Patch-first editing with strong concurrency guards.
- Explicit full-file writes only when full replacement is the right operation.
- Shared safety, validation, and output semantics across every file tool.

This document is executable planning material. Each item should be directly
convertible into implementation tasks, tests, and documentation updates.

## Current Baseline

Current Alpha file tools live under `src/alpha_agent/tools/files.py` and are
registered from `src/alpha_agent/tools/default.py`.

Implemented tools:

- `file_list`: list directory entries under `tools.files.allowed_roots`.
- `file_read`: read UTF-8 text files with optional line range and max chars.
- `file_search`: case-insensitive literal substring search in UTF-8 text files.
- `file_patch`: apply structured line-range edits under `tools.files.write_roots`
  when `tools.files.patch_enabled` is true and write roots are configured.

Baseline strengths to preserve:

- Read-only tools are enabled by default; write tools are separately gated.
- Read roots and write roots are separate.
- Existing-file writes require `expected_sha256`, which is stronger than a
  timestamp-only stale-write guard.
- Binary files, NUL bytes, symlink targets, symlink ancestors, and out-of-root
  paths are rejected.
- Tool results are compact and bounded.

Baseline gaps:

- Search is literal substring only; no regex, file-only mode, count mode,
  pagination, file type filter, multiline mode, or mtime ranking.
- `file_list` mixes directory listing with glob-like discovery; there is no
  first-class "find files by name" tool.
- `file_read` lacks offset/limit pagination, line-number formatted output,
  repeated-read dedup, read-loop guard, similar-path suggestions, and device
  file blocking.
- `file_patch` only supports line-range edits; it does not support
  old/new-string edits, replace-all, or multi-file patch text.
- Writes are direct `write_bytes`; there is no temp-rename or readback
  verification layer.
- There is no shared read/write state registry, per-path lock, post-write lint,
  LSP diagnostics, or file-history snapshot.
- No delete, move, or copy tools exist.

## Reference Patterns To Adopt

Hermes patterns worth adopting:

- `search_files` supports content search and file search in one interface, with
  regex, glob filters, output modes, context, offset, and limit.
- File search is ripgrep-first, fallback-capable, and sorted by modification
  time when possible.
- Read tools add line numbers, enforce pagination limits, suggest similar files,
  block device paths, deduplicate repeated reads, and stop repeated read/search
  loops.
- Write and patch tools use per-path locks and stale-read warnings for
  concurrent subagents.
- Patch supports both old/new-string replacement and multi-file patch text.
- Post-write validation reports syntax/lint/LSP errors introduced by the edit,
  not unrelated pre-existing errors.

Claude Code patterns worth adopting:

- Separate fast file discovery (`Glob`) from content search (`Grep`), even if
  Alpha keeps a smaller external surface.
- Search supports `output_mode`, `head_limit`, `offset`, `type`, `glob`,
  context flags, case-insensitive mode, and multiline regex.
- Read supports token-aware limits for bounded text views.
- Existing files must be read before overwrite/edit in stateful tools; Alpha can
  keep hash-based guards instead, but should still track reads for better
  warnings and dedup.
- Edit supports `old_string`, `new_string`, and `replace_all`, with exact unique
  matching before applying changes.
- Write/Edit update file history and notify diagnostics systems after writes.

Codex-style patterns worth adopting:

- Prefer patch over full-file write for existing files.
- Use an explicit patch-text grammar for multi-file changes:
  `*** Begin Patch`, `*** Add File`, `*** Update File`, and `*** End Patch`.
- Make patch parsing strict and deterministic; reject ambiguous or malformed
  patches before touching disk.
- Defer delete and move grammar until destructive file operations are explicitly
  accepted as a separate scope.

## Target Toolset

The target external toolset should stay small:

| Tool | Purpose | Read-only | Write-gated |
| --- | --- | --- | --- |
| `file_glob` | Find files by path/name pattern. | Yes | No |
| `file_search` | Search file contents. | Yes | No |
| `file_read` | Read bounded text content. | Yes | No |
| `file_patch` | Targeted edits and multi-file patch text. | No | Yes |
| `file_write` | Create or intentionally replace whole files. | No | Yes |

Do not add `file_delete`, `file_move`, or `file_copy` until the core workflow is
stable. They are useful, but they expand the destructive surface before the
read/search/edit loop is mature.

## Shared Contracts

Every file tool should follow these contracts.

Path policy:

- All paths are resolved relative to configured roots when not absolute.
- Read tools must stay under `tools.files.allowed_roots`.
- Write tools must stay under `tools.files.write_roots`.
- Symlink final targets and symlink ancestors are rejected for write tools.
- Device paths that can block or produce infinite output are rejected before any
  read attempt. This is defense-in-depth: root confinement already excludes them
  unless `allowed_roots` is widened to include a device directory.
- NUL bytes in path or content are rejected.

Output policy:

- Results must be bounded by `tools.files.max_output_chars`.
- Any truncation must return `truncated: true`.
- Paginated tools must return enough data for the next call:
  `limit`, `offset`, `next_offset`, and `total_count` when available.
- Paths in output should be display paths relative to the relevant root.
- Errors should be deterministic and specific enough for the model to correct
  its next call.

Concurrency policy:

- Single-file writes in `range` and `replace` mode require `expected_sha256` for
  existing files.
- `patch_text` mode carries no per-file `expected_sha256`. Its concurrency guard
  is strict context-line matching: if the surrounding context in the patch no
  longer matches the file on disk, the operation fails before any write. Do not
  require `expected_sha256` for `patch_text`.
- Internal state may add warnings, but must not replace hash or context-match
  validation.
- Writes to the same resolved path are serialized with a per-path lock.
- Multi-file patch locks are acquired in sorted path order.
- After a successful write, all read dedup entries for affected paths are
  invalidated.

Validation policy:

- Read and search skip unsupported binary files unless the tool explicitly
  supports that format.
- Write and patch perform post-write verification by reading the bytes back.
- Syntax checks are run for formats with cheap deterministic parsers:
  Python, JSON, TOML, YAML if dependency support exists.
- LSP diagnostics are optional and should be an enrichment layer, never a write
  blocker unless a future policy explicitly says otherwise.

Turn-scoped tool state:

- Dedup and loop-guard state lives for exactly one turn, where a turn is the full
  processing loop for a single user message (`_run_agent_loop`) and may span many
  tool-call iterations. A fresh turn starts with empty state, so there is no
  cross-turn leakage and no manual reset.
- Neither current location works, and fixing this is a prerequisite for Phase 2:
  - `ToolExecutionContext.extensions` is rebuilt on every tool-call iteration in
    `_execute_tool_calls`, so it cannot carry state across the iterations of one
    turn.
  - Tool instances are session-scoped (one agent and registry per session), so
    instance attributes would leak counters across turns.
- Target design (chosen for overall fit, not minimal change):
  - The runtime owns a mutable `TurnToolState` (proposed name) created once per
    turn, as a sibling of the existing `AgentTurnContext` identity record.
    `AgentTurnContext` is frozen and stays an identity-only record; the mutable
    state is a separate object with the same lifecycle, threaded
    `_run_agent_loop` -> `_execute_tool_calls` -> `ToolExecutor.execute` -> each
    `ToolExecutionContext`.
  - Surface it to tools as a typed `ToolExecutionContext` field, not through the
    stringly-typed `extensions` bag. A typed field is mypy-checked and
    discoverable; the existing `extensions` memory contexts are the legacy pattern
    and can migrate onto the same field later.
- Split the two concerns by altitude:
  - The repeated-call loop guard is generic (identical `tool_name` plus canonical
    arguments repeated within a turn). Implement it once in `ToolExecutor` against
    `TurnToolState` so it protects every tool and can short-circuit before
    dispatch: warn on the third identical call, block on the fourth. It is always
    on, with no config toggle.
  - Read/search dedup is file-tool-specific (content identity = resolved path,
    offset, limit, mtime; returns a content stub). It lives in the file tools but
    records its ledger in the same `TurnToolState`.

## Tool Contracts

### `file_glob`

Purpose: find files by name or path pattern without reading file contents.

Parameters:

- `pattern: string` - glob pattern such as `*.py`, `**/*test*.py`, or
  `src/**/config*.toml`.
- `path?: string` - root directory to search; defaults to `.` under the first
  allowed root.
- `max_depth?: integer` - limit traversal depth below `path`; `max_depth: 1`
  lists only immediate children of `path`, replacing the non-recursive listing
  that `file_list` provided.
- `limit?: integer` - max returned paths.
- `offset?: integer` - pagination offset.
- `sort?: "mtime_desc" | "path_asc"` - default `mtime_desc`.
- `include_dirs?: boolean` - default false.

Output:

- `path`
- `pattern`
- `files: [{ path, type, size, mtime }]`
- `total_count` when available cheaply
- `limit`
- `offset`
- `next_offset`
- `truncated`

Implementation tasks:

- Prefer ripgrep `rg --files` when available; pass `--max-depth` when `max_depth`
  is set.
- Fall back to Python traversal using existing root and excluded-dir logic.
- Respect current excluded directories by default.
- Cover the single-directory browse case (`max_depth: 1` with `include_dirs:
  true`) so removing `file_list` does not lose directory listing.
- Sort by `mtime_desc` for relevance, with a stable path tiebreaker. Note that
  ripgrep's `--sort`/`--sortr modified` disables parallelism; sorting in Python
  after a bounded collection may be preferable for large trees.
- `total_count` is often not cheaply available from a streaming `rg` run; return
  it only when the fallback enumerates fully, and omit it otherwise.
- Add tests for pagination, root policy, hidden/excluded directories, symlinks,
  single-directory listing, and deterministic sorting.

### `file_search`

Purpose: search file contents with bounded, paginated output.

Parameters:

- `pattern: string` - regex by default.
- `mode?: "regex" | "literal"` - default `regex`.
- `path?: string` - file or directory to search.
- `glob?: string` - file glob filter.
- `type?: string` - optional ripgrep type filter when using rg.
- `output_mode?: "content" | "files_with_matches" | "count"` - default
  `content`.
- `case_sensitive?: boolean` - default false.
- `context?: integer` - symmetric context lines.
- `before_context?: integer`
- `after_context?: integer`
- `limit?: integer`
- `offset?: integer`
- `multiline?: boolean` - default false.

Output for `content`:

- `matches: [{ path, line_number, line, before, after }]`
- `match_count` - total matches when available cheaply, otherwise returned
  match count.
- `limit`
- `offset`
- `next_offset`
- `truncated`

Output for `files_with_matches`:

- `files: [{ path, size, mtime }]`
- `file_count`
- `limit`
- `offset`
- `next_offset`
- `truncated`

Output for `count`:

- `counts: [{ path, count }]`
- `total_matches`
- `file_count`
- `limit`
- `offset`
- `next_offset`
- `truncated`

Implementation tasks:

- Replace the current `query`, `max_matches`, and `context_lines` schema with
  the target `pattern`, `limit`, and context fields in one direct contract
  change.
- Use ripgrep when available for regex, multiline, type filters, and speed.
- Keep a Python fallback for literal mode and environments without rg.
- Make `literal` mode escape user input before passing it to rg.
- Rely on the generic `ToolExecutor` repeated-call guard (see the Turn-scoped
  tool state contract) for repeated identical searches; do not add a search-only
  loop counter inside this tool.
- Add tests for regex, literal escaping, output modes, context, pagination,
  binary skip, missing path suggestion, and max output truncation.

### `file_read`

Purpose: read a bounded view of a text file.

Parameters:

- `path: string`
- `offset?: integer` - 1-based starting line; default 1.
- `limit?: integer` - number of lines; default from config.
- `max_chars?: integer` - hard character cap after line selection.
- `format?: "plain" | "line_numbered"` - default `line_numbered`.

Output:

- `path`
- `content`
- `offset`
- `limit`
- `returned_lines`
- `total_lines`
- `size`
- `sha256`
- `truncated`
- `next_offset`
- `format`

Implementation tasks:

- Replace `start_line/end_line` with `offset/limit` in the target contract.
  The project does not require compatibility with old tool schemas.
- Add line-numbered output so model edits can cite stable line ranges.
- Treat line-numbered output as display-only: `format: "line_numbered"` content
  must never be fed back as `file_write` content or as a `file_patch` replace
  `old_string`, which match raw file text. Models copying text into a replace
  edit should read with `format: "plain"`.
- Add device path blocklist.
- Add file-not-found suggestions from nearby directory entries and likely cwd
  mistakes.
- Add repeated-read dedup keyed by resolved path, offset, limit, and mtime,
  recorded in `TurnToolState` (see the Turn-scoped tool state contract). Do not
  store it on tool instances or in the per-iteration
  `ToolExecutionContext.extensions`.
- Rely on the generic `ToolExecutor` repeated-call guard for repeated-read loops;
  do not add a read-only loop counter inside this tool.
- Add tests for beyond-EOF reads, empty files, truncation, hashes, dedup,
  stale mtime invalidation, and device path rejection.

### `file_patch`

Purpose: targeted changes to existing or new text files. This remains the
primary write tool.

Parameters:

- `mode: "range" | "replace" | "patch_text"`
- `expected_sha256?: string` - `range` and `replace` mode only; required for
  existing files in those modes and ignored in `patch_text` mode.
- `create_if_missing?: boolean` - `range` and `replace` mode only; in
  `patch_text` mode new files are expressed with `*** Add File`.

Range mode:

- `path: string`
- `edits: [{ start_line, end_line, replacement }]`

Replace mode:

- `path: string`
- `old_string: string`
- `new_string: string`
- `replace_all?: boolean`

Patch-text mode:

- `patch: string`

Patch-text grammar:

```text
*** Begin Patch
*** Add File: path/to/new_file.py
+new content
*** Update File: path/to/existing.py
@@ optional context @@
 old context
-removed line
+added line
*** End Patch
```

Output:

- `files_modified`
- `files_created`
- `before_sha256` and `after_sha256` for single-file operations
- `diff`
- `bytes_written`
- `applied_edits`
- `warnings`
- `validation`

Implementation tasks:

- Keep current range mode behavior and move shared write helpers behind an
  internal writer module.
- Add replace mode with exact unique matching first.
- Reject replace mode when `old_string` is absent.
- Reject replace mode when multiple matches exist and `replace_all` is false.
- Add optional fuzzy matching only after exact mode is tested and stable.
- Add patch-text parser with strict grammar and no partial application.
- Use strict context-line matching as the `patch_text` concurrency guard: reject
  the patch when on-disk context no longer matches, rather than requiring a hash.
- For multi-file patch text, validate every operation before writing any file.
- Lock all affected files in sorted resolved-path order.
- Write via temp file and atomic rename where supported.
- Verify by reading back written content.
- Run syntax validation and include diagnostics in result.
- Add tests for exact replace, replace-all, no-match, ambiguous match,
  multi-file add/update, malformed patch, lock ordering, and rollback on
  validation failure.

### `file_write`

Purpose: create new files or intentionally replace an entire file when patch is
the wrong abstraction.

Parameters:

- `path: string`
- `content: string`
- `expected_sha256?: string`
- `create_if_missing?: boolean` - default true.
- `overwrite?: boolean` - default false for existing files.

Rules:

- Existing file replacement requires `overwrite: true`.
- Existing file replacement requires `expected_sha256`.
- New file creation rejects non-empty `expected_sha256`.
- Parent directory creation should be explicit:
  `create_parent_dirs?: boolean`, default false.
- The tool should return a diff for updates, not the full written content.

Implementation tasks:

- Add only after `file_patch` has replace mode and atomic write helpers.
- Reuse the same path, binary, NUL, hash, lock, and validation helpers as
  `file_patch`.
- Run post-write verification and syntax validation.
- Add tests for create, overwrite rejection, hash mismatch, parent directory
  behavior, max file size, and diff output bounds.

## Internal Modules To Extract

The current single-file implementation should be split when the next feature
lands, not after it becomes harder to unwind.

Because the current implementation is `src/alpha_agent/tools/files.py`, the
first extraction should convert it into a `src/alpha_agent/tools/files/` package
with an `__init__.py` that exports the tool classes and constants used by the
registry and tests.

Suggested modules:

- `alpha_agent.tools.files.paths`
  - root normalization
  - display path rendering
  - read-root and write-root resolution
  - symlink and device path policy
- `alpha_agent.tools.files.reading`
  - bounded byte reads
  - text decoding
  - line slicing
  - line-number formatting
  - hash calculation
- `alpha_agent.tools.files.searching`
  - rg invocation
  - Python fallback
  - output mode mapping
  - pagination helpers
- `alpha_agent.tools.files.patching`
  - range edit parser
  - replace edit parser
  - patch-text parser
  - diff generation
- `alpha_agent.tools.files.writing`
  - hash checks
  - per-path locks
  - temp write and rename
  - readback verification
  - dedup invalidation
- `alpha_agent.tools.files.validation`
  - binary detection
  - syntax checks
  - optional diagnostics adapters
- `alpha_agent.tools.files.state`
  - read dedup
  - repeated read/search loop guard
  - read stamps
  - cross-agent stale warnings

Extraction trigger:

- Extract as soon as the same helper logic appears in three tools or three test
  groups.
- Do not keep path resolution, binary checks, or hash validation duplicated
  across tool classes.

## Configuration

Keep current root settings:

- `tools.files.enabled`
- `tools.files.allowed_roots`
- `tools.files.patch_enabled`
- `tools.files.write_roots`
- `tools.files.max_read_chars`
- `tools.files.max_file_bytes`
- `tools.files.max_output_chars`

Replace current limit settings directly:

- `tools.files.max_search_matches` -> `tools.files.max_search_results`
- `tools.files.max_list_entries` -> `tools.files.max_glob_results`

Add:

- `tools.files.max_read_lines`
- `tools.files.read_dedup_enabled`
- `tools.files.post_write_validation_enabled`
- `tools.files.create_parent_dirs_enabled`

Do not add a `use_ripgrep` configuration toggle. Ripgrep is an implementation
backend: auto-detect it when available and fall back to the Python traversal or
search path when it is not.

Do not add a general "allow destructive file operations" flag. Each destructive
tool should be individually gated by tool availability and write roots.

Each config key change touches several sites in `src/alpha_agent/config.py`:
`DEFAULT_CONFIG_TOML`, `CONFIG_KEY_TYPES`, the relevant `*_INT_CONFIG_KEYS` set,
the `FileToolConfig` dataclass, `_file_tool_config`, `_validate_loaded_config`,
and `_validate_config_data`. New keys also need an `ALPHA_FILE_TOOL_*` env
override and a `.env.example` entry. The two renames are breaking config changes;
the non-goals accept this, so no migration shim is required.

## Delivery Plan

### Phase 1: Search And Discovery

Deliverables:

- Add `file_glob`.
- Replace the registered external `file_list` surface with `file_glob`, after
  confirming `file_glob` covers single-directory listing (`max_depth: 1` with
  `include_dirs: true`) so directory browsing is not lost.
- Upgrade `file_search` to regex plus output modes and pagination.
- Add rg backend with Python fallback.
- Update README built-in tools section.

Acceptance:

- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest tests/test_file_tools.py -q` (keep file-tool tests runnable as a
  group even if later split into focused modules)
- New tests cover file-list replacement, regex, literal mode, output modes,
  pagination, and glob sorting.

### Phase 2: Read Ergonomics And State

Prerequisite:

- Add the turn-scoped tool state plumbing (see the Turn-scoped tool state
  contract): a runtime-owned `TurnToolState` created per turn and surfaced as a
  typed `ToolExecutionContext` field, plus the generic repeated-call loop guard in
  `ToolExecutor`. Dedup and the read/search loop guard depend on this.

Deliverables:

- Change `file_read` contract to `offset/limit`.
- Add line-numbered output.
- Add device path blocklist.
- Add similar-path suggestions.
- Add read/search dedup in the file tools, recorded in `TurnToolState`.

Acceptance:

- Existing read tests are rewritten to target `offset/limit`.
- Dedup test proves unchanged repeated reads return a stub.
- Stale read test proves modified files return fresh content.
- Loop guard test proves repeated identical reads/searches within one turn stop
  wasting iterations, and that counters reset on a new turn.

### Phase 3: Patch As The Primary Edit Tool

Deliverables:

- Refactor shared write helpers.
- Add `file_patch mode=replace`.
- Add atomic write and readback verification.
- Add cheap syntax validation for Python, JSON, TOML, and YAML when available.

Acceptance:

- Replace mode refuses ambiguous matches unless `replace_all` is true.
- Hash mismatch still blocks writes before disk mutation.
- Write verification failure is surfaced as an error.
- Syntax validation appears in tool output and does not hide the diff.

### Phase 4: Multi-file Patch Text

Deliverables:

- Add `file_patch mode=patch_text`.
- Support add and update operations.
- Defer delete and move operations until after `file_patch` and `file_write`
  share stable destructive-operation safety infrastructure.
- Validate the full patch before writing anything.
- Lock all affected files in stable order.

Acceptance:

- Multi-file patches are all-or-nothing.
- Malformed patch text does not touch disk.
- A patch that targets outside write roots fails before touching disk.
- Diffs are bounded and identify every changed file.

### Phase 5: Full-file Write

Deliverables:

- Add `file_write`.
- Restrict it to explicit create or explicit full overwrite.
- Require `expected_sha256` for existing-file overwrite.
- Reuse validation, locking, verification, and diff output.

Acceptance:

- Existing file overwrite without `overwrite: true` fails.
- Existing file overwrite without matching `expected_sha256` fails.
- New file creation with non-empty `expected_sha256` fails.
- Result returns create/update metadata and bounded diff.

## Testing Matrix

Every file tool change should update `tests/test_file_tools.py` or split file
tool tests into focused modules.

Required categories:

- Root policy: relative paths, absolute paths, outside-root paths.
- Symlinks: final symlink, ancestor symlink, broken symlink.
- Binary/NUL: binary extension, NUL bytes in file, NUL bytes in replacement.
- Bounds: max bytes, max chars, max lines, max result size.
- Pagination: limit, offset, next offset, truncation.
- Search: regex, literal, glob filter, count mode, files mode, context mode.
- Write safety: expected hash, stale content, lock behavior, readback
  verification.
- Patch parsing: malformed inputs, overlapping edits, ambiguous replace,
  multi-file all-or-nothing.
- Validation: syntax success, syntax failure, validation skipped.

CI gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## Documentation Updates Per Phase

Each phase must update:

- `README.md` built-in tools section.
- `config.example.toml` if configuration changes.
- `.env.example` if environment overrides change.
- Tool descriptions in `ToolSpec` so the model receives the new contract.
- This roadmap if scope changes materially.

## Explicit Non-goals

- Do not expose unrestricted filesystem write access.
- Do not make shell commands the primary file editing path.
- Do not add delete/move/copy before patch and write have shared safety
  infrastructure.
- Do not support arbitrary binary editing through these text tools.
- Do not include image, PDF, or notebook support in this roadmap; revisit those
  in a separate plan if rich-format workflows become a priority.
- Do not add compatibility shims for old file tool schemas unless a real
  external consumer requires it.
- Do not redact configured secret values from read or search output. This was
  considered and rejected: redaction risks corrupting legitimate file content
  (config samples, fixtures, docs) and degrading tool accuracy. Secret hygiene
  stays an operator concern enforced through `allowed_roots` scoping.
