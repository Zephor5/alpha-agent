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
- File search is ripgrep-first and sorted by modification time when possible.
- Read tools add line numbers, enforce pagination limits, suggest similar files,
  block device paths, deduplicate repeated reads, and stop repeated tool-call
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

Reference decisions for Alpha:

- `rg` is the required backend for content search and recursive file discovery.
  There is no Python content-search fallback in this roadmap.
- Directory entry listing is a separate local operation: `file_glob` may use
  Python `Path.iterdir` for `max_depth: 1` browse calls and for returning
  directories when `include_dirs: true`, because `rg --files` only returns files.
- Hash checks and patch context matching are hard write guards. Runtime state
  may add warnings and dedup, but it must not replace those guards.
- Syntax and LSP diagnostics are advisory output. They should never cause a
  successful disk write to be rolled back.

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

Backend availability policy:

- `file_search` requires `rg` for all modes.
- Recursive file-only `file_glob` discovery requires `rg`.
- Single-directory browse and directory inclusion in `file_glob` use local
  directory entry APIs and do not require `rg`.
- Do not add a `use_ripgrep` toggle. Backend availability should be explicit in
  the tool result when a requested operation requires `rg`.

Concurrency policy:

- Single-file writes in `range` and `replace` mode require `expected_sha256` for
  existing files.
- `patch_text` mode carries no per-file `expected_sha256`. Its concurrency guard
  is strict context-line matching: if the surrounding context in the patch no
  longer matches the file on disk, the operation fails before any write. Do not
  require `expected_sha256` for `patch_text`.
- Internal state may add warnings, but must not replace hash or context-match
  validation.
- Writes to the same resolved path are serialized with a process-wide per-path
  lock registry, not a lock stored on a tool instance.
- Multi-file patch locks are acquired in sorted path order.
- After a successful write, all read dedup entries for affected paths are
  invalidated.

Validation policy:

- Read and search skip unsupported binary files unless the tool explicitly
  supports that format.
- Write and patch perform post-write verification by reading the bytes back.
- Patch parse, path, root, hash, and context-match failures happen before writes
  and leave disk untouched.
- Syntax checks are run for formats with cheap deterministic parsers:
  Python, JSON, TOML, YAML if dependency support exists.
- LSP diagnostics are optional and should be an enrichment layer, never a write
  blocker unless a future policy explicitly says otherwise.
- Syntax and LSP diagnostics are not rollback triggers.

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
    `TurnToolState` so it protects every tool: allow the third identical call but
    include a model-visible warning in the tool result; block the fourth before
    dispatch with a structured error. It is always on, with no config toggle.
  - Read dedup is file-tool-specific (content identity = resolved path, offset,
    limit, `max_chars`, `format`, file size, and mtime; returns a content stub).
    It lives in the file tools but records its ledger in the same
    `TurnToolState`.

## Tool Contracts

### `file_glob`

Purpose: find files by name or path pattern without reading file contents.

Parameters:

- `pattern?: string` - glob pattern such as `*.py`, `**/*test*.py`, or
  `src/**/config*.toml`; defaults to `*`.
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

- Use ripgrep `rg --files` for recursive file discovery; pass `--max-depth` when
  `max_depth` is set.
- Return a deterministic unavailable/error result when a recursive file
  discovery request requires `rg` and `rg` is unavailable.
- Use local directory entry APIs for the single-directory browse case and for
  directory results when `include_dirs: true`; this is directory-listing
  behavior, not a recursive file-search fallback.
- Respect current excluded directories by default.
- Cover the single-directory browse case (`max_depth: 1` with `include_dirs:
  true`) so removing `file_list` does not lose directory listing.
- Sort by `mtime_desc` for relevance, with a stable path tiebreaker. Note that
  ripgrep's `--sort`/`--sortr modified` disables parallelism; sorting in Python
  after a bounded collection may be preferable for large trees.
- `total_count` is often not cheaply available from a streaming `rg` run; return
  it only when the implementation fully enumerates results cheaply, and omit it
  otherwise.
- Add tests for pagination, root policy, hidden/excluded directories, symlinks,
  single-directory listing, deterministic sorting, and `rg` unavailable behavior.

### `file_search`

Purpose: search file contents with bounded, paginated output.

Parameters:

- `pattern: string` - regex by default.
- `mode?: "regex" | "literal"` - default `regex`.
- `path?: string` - file or directory to search.
- `glob?: string` - file glob filter.
- `type?: string` - optional ripgrep type filter.
- `output_mode?: "content" | "files_with_matches" | "count"` - default
  `files_with_matches`.
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
- Require ripgrep for all content search modes. Return a deterministic
  unavailable/error result when `rg` is unavailable.
- Use ripgrep for regex, fixed-string literal mode, multiline, type filters,
  output modes, and pagination.
- Use rg fixed-string mode for `literal` rather than hand-escaped regex.
- Preserve `tools.files.max_file_bytes` by passing an equivalent rg max-filesize
  limit or by filtering candidate files before search.
- Rely on the generic `ToolExecutor` repeated-call guard (see the Turn-scoped
  tool state contract) for repeated identical searches; do not add a search-only
  loop counter inside this tool.
- Add tests for `rg` unavailable behavior, regex, literal fixed-string search,
  output modes, context, pagination, binary skip, missing path suggestion,
  max-filesize handling, type filters, multiline mode, and max output truncation.

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
- Add repeated-read dedup keyed by resolved path, offset, limit, `max_chars`,
  `format`, file size, and mtime, recorded in `TurnToolState` (see the
  Turn-scoped tool state contract). Do not store it on tool instances or in the
  per-iteration
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

Patch-text grammar rules:

- The patch must contain exactly one `*** Begin Patch` header and one
  `*** End Patch` footer.
- Only `*** Add File` and `*** Update File` operations are accepted in this
  roadmap; delete and move operations are rejected.
- Patch paths must be project-root or write-root relative display paths, never
  machine-specific absolute paths.
- `Add File` fails if the target already exists. `Update File` fails if the
  target does not exist.
- Multiple operations against the same resolved path are rejected in the first
  implementation.
- Update hunks require exact context matching. Ambiguous or missing context
  fails before any file is written.

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
- Add patch-text parser with strict grammar and no partial application.
- Do not add fuzzy matching in this roadmap; exact context and exact string
  matching are the default edit behavior.
- Use strict context-line matching as the `patch_text` concurrency guard: reject
  the patch when on-disk context no longer matches, rather than requiring a hash.
- For multi-file patch text, validate every operation before writing any file.
- Lock all affected files in sorted resolved-path order.
- Write via temp file and atomic rename where supported.
- Verify by reading back written content.
- Run advisory syntax validation and include diagnostics in result.
- Add tests for exact replace, replace-all, no-match, ambiguous match,
  multi-file add/update, malformed patch, lock ordering, advisory validation,
  and no disk mutation on pre-write parse/root/context failures.

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
- `create_parent_dirs: true` only succeeds when
  `tools.files.create_parent_dirs_enabled` is enabled.
- The tool should return a diff for updates, not the full written content.

Implementation tasks:

- Add only after `file_patch` has replace mode and atomic write helpers.
- Reuse the same path, binary, NUL, hash, lock, and validation helpers as
  `file_patch`.
- Run post-write verification and advisory syntax validation.
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
  - output mode mapping
  - pagination helpers
- `alpha_agent.tools.files.patching`
  - range edit parser
  - replace edit parser
  - patch-text parser
  - diff generation
- `alpha_agent.tools.files.writing`
  - hash checks
  - process-wide per-path locks
  - temp write and rename
  - readback verification
  - dedup invalidation
- `alpha_agent.tools.files.validation`
  - binary detection
  - syntax checks
  - optional diagnostics adapters
- `alpha_agent.tools.files.state`
  - read dedup
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
- `tools.files.create_parent_dirs_enabled`

Do not add a `use_ripgrep` configuration toggle. Ripgrep is a required backend
for `file_search` and recursive file-only `file_glob` discovery. If it is
unavailable, those calls return deterministic unavailable/error results.
Directory listing behavior in `file_glob` remains local and does not require
`rg`.

Do not add a general "allow destructive file operations" flag. Each destructive
tool should be individually gated by tool availability and write roots.

Each config key change touches several sites in `src/alpha_agent/config.py`:
`DEFAULT_CONFIG_TOML`, `CONFIG_KEY_TYPES`, the relevant `*_INT_CONFIG_KEYS` set,
the `FileToolConfig` dataclass, `_file_tool_config`, `_validate_loaded_config`,
and `_validate_config_data`. New keys also need an `ALPHA_FILE_TOOL_*` env
override and a `.env.example` entry. The two renames are breaking config changes;
the non-goals accept this, so no migration shim is required.

## Delivery Plan

Phases are ordered by dependency and implementation size. A phase may use local
smoke tests or focused tests while it is being implemented, but the formal
acceptance gate is the final "Overall Completion Acceptance" section. Do not
force every phase to be independently shippable if doing so creates artificial
compatibility work.

Dependency shape:

- Phase 1 is the shared file-tool foundation.
- Phases 2 and 3 both depend on Phase 1 and may be implemented in either order,
  but should land sequentially because both touch tool registration and file-tool
  tests.
- Phase 4 is runtime state plumbing and must land before read dedup.
- Phase 5 depends on Phases 1 and 4.
- Phase 6 is the write-safety foundation.
- Phases 7, 8, and 9 depend on Phase 6. Keep `file_patch` improvements before
  `file_write`, because patch remains the primary edit path.

### Phase 1: File Tool Foundation

Purpose: create the shared structure needed by discovery, search, read, and
write changes without changing the external toolset more than necessary.

Deliverables:

- Convert `src/alpha_agent/tools/files.py` into a
  `src/alpha_agent/tools/files/` package with an `__init__.py` registry export.
- Extract path resolution, root policy, display path rendering, binary/NUL
  checks, output bounding, and common result helpers used by current tools.
- Add an `rg` capability helper and a deterministic unavailable/error result
  helper for operations that require `rg`.
- Apply direct config renames/additions:
  `max_search_results`, `max_glob_results`, `max_read_lines`, and
  `create_parent_dirs_enabled`, with no migration shim.
- Update config documentation files for the renamed and added keys.

### Phase 2: File Discovery

Purpose: replace directory listing and path discovery with the target
`file_glob` surface.

Deliverables:

- Add `file_glob` with optional `pattern`, pagination, `sort`, `max_depth`, and
  `include_dirs`.
- Use `rg --files` for recursive file discovery and local directory entry APIs
  for single-directory browse and directory inclusion.
- Preserve root, symlink, excluded-directory, binary, output-bound, and display
  path policies through the shared helpers.
- Replace the registered external `file_list` surface with `file_glob` after
  proving `max_depth: 1` with `include_dirs: true` covers directory browsing.
- Add focused tests for list replacement, pagination, sorting, excluded paths,
  and deterministic `rg` unavailable behavior.

### Phase 3: Content Search

Purpose: upgrade `file_search` into an rg-backed content search tool with a
clear contract.

Deliverables:

- Replace `query`, `max_matches`, and `context_lines` with `pattern`, `limit`,
  `offset`, output modes, context fields, `glob`, `type`, and `multiline`.
- Require `rg` for all content search modes.
- Implement regex, fixed-string literal mode, files-with-matches, count mode,
  content mode, context lines, pagination, and max-filesize handling.
- Keep repeated-search protection in the generic loop guard, not inside
  `file_search`.
- Add focused tests for regex, literal fixed-string search, output modes,
  context, pagination, type filters, multiline, truncation, and `rg` unavailable
  behavior.

### Phase 4: Turn-Scoped Tool State

Purpose: add the runtime state lifecycle needed for dedup and generic loop
protection.

Deliverables:

- Add a runtime-owned `TurnToolState` created once per `_run_agent_loop`.
- Surface the state as a typed `ToolExecutionContext` field and thread it through
  `_execute_tool_calls`, `ToolExecutor.execute`, and tool execution.
- Implement the generic repeated-call guard in `ToolExecutor`: warn in the third
  identical call result and block the fourth with a structured error.
- Keep the guard always on with no config toggle.
- Add focused runtime tests proving state persists across tool-call iterations
  within a turn and resets between turns.

### Phase 5: File Read Ergonomics

Purpose: make reads line-addressable, bounded, and state-aware.

Deliverables:

- Change `file_read` to `offset` and `limit`.
- Add `plain` and `line_numbered` formats, with line-numbered output documented
  as display-only for edits.
- Add device path blocking and file-not-found suggestions.
- Add repeated-read dedup recorded in `TurnToolState` with the full read identity
  key: resolved path, offset, limit, `max_chars`, `format`, size, and mtime.
- Rewrite focused read tests for pagination, hashes, truncation, dedup, stale
  invalidation, suggestions, and device path rejection.

### Phase 6: Write Safety Foundation

Purpose: centralize write mechanics before adding new write modes.

Deliverables:

- Extract writer helpers for hash checks, process-wide per-path locks, sorted
  multi-file lock acquisition, temp write, atomic rename where supported,
  readback verification, bounded diff output, and read dedup invalidation.
- Add advisory syntax validation helpers for Python, JSON, TOML, and YAML when
  dependency support exists.
- Refactor current `file_patch` range mode onto the shared writer helpers without
  changing its target external behavior beyond improved verification output.
- Add focused write-safety tests for hash mismatch, lock ordering helpers,
  readback verification errors, bounded diffs, and advisory validation output.

### Phase 7: Replace Patch Mode

Purpose: add the first new edit mode on top of the shared write-safety layer.

Deliverables:

- Add `file_patch mode=replace` with `old_string`, `new_string`, and
  `replace_all`.
- Require exact unique matching unless `replace_all` is true.
- Reject absent `old_string` and ambiguous matches before disk mutation.
- Preserve `expected_sha256` for existing-file replace operations.
- Add focused tests for exact replace, replace-all, no-match, ambiguous match,
  hash mismatch, and diff output bounds.

### Phase 8: Multi-file Patch Text

Purpose: add strict Codex-style patch text for multi-file changes.

Deliverables:

- Add `file_patch mode=patch_text` with strict `*** Begin Patch`,
  `*** Add File`, `*** Update File`, and `*** End Patch` grammar.
- Support add and update operations only; reject delete, move, duplicate paths,
  absolute paths, malformed patches, and missing or ambiguous context.
- Validate every file operation before writing anything.
- Lock all affected files in sorted resolved-path order.
- Add focused tests for multi-file all-or-nothing behavior, malformed input,
  outside-root rejection, context mismatch, duplicate paths, and bounded per-file
  diffs.

### Phase 9: Full-file Write

Purpose: add the explicit whole-file write path after patch editing is mature.

Deliverables:

- Add `file_write`.
- Restrict it to explicit create or explicit full overwrite.
- Require `overwrite: true` and matching `expected_sha256` for existing-file
  overwrite.
- Allow parent directory creation only when both `create_parent_dirs: true` and
  `tools.files.create_parent_dirs_enabled` are set.
- Reuse path, hash, locking, verification, advisory validation, and bounded diff
  helpers from the write-safety layer.
- Add focused tests for create, overwrite rejection, hash mismatch, parent
  directory behavior, max file size, and bounded diff output.

### Overall Completion Acceptance

- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`
- Registered file tools expose the target toolset:
  `file_glob`, `file_search`, `file_read`, `file_patch`, and `file_write`.
- Tests cover the Testing Matrix categories below, including deterministic `rg`
  unavailable behavior, turn-state reset, read dedup, and write no-mutation
  guarantees for pre-write failures.
- `README.md`, `config.example.toml`, `.env.example`, and `ToolSpec`
  descriptions reflect the final contracts.

## Testing Matrix

Every file tool change should update `tests/test_file_tools.py` or split file
tool tests into focused modules.

Required categories:

- Root policy: relative paths, absolute paths, outside-root paths.
- Symlinks: final symlink, ancestor symlink, broken symlink.
- Binary/NUL: binary extension, NUL bytes in file, NUL bytes in replacement.
- Bounds: max bytes, max chars, max lines, max result size.
- Pagination: limit, offset, next offset, truncation.
- Search: `rg` unavailable behavior, regex, fixed-string literal, glob filter,
  count mode, files mode, context mode, multiline mode, type filters,
  max-filesize handling.
- Write safety: expected hash, stale content, lock behavior, readback
  verification.
- Patch parsing: malformed inputs, overlapping edits, ambiguous replace,
  multi-file all-or-nothing, no disk mutation on pre-write failures.
- Validation: syntax success, syntax failure, validation skipped, diagnostics
  returned without rollback.

CI gate:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## Documentation Updates

Any implementation slice that changes public tool behavior, configuration, or
model-facing tool descriptions must update the relevant active docs in the same
change. The final overall acceptance still checks that all docs match the final
tool contracts.

Relevant docs:

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
- Do not add Python content-search fallback for `rg`-backed search and recursive
  file discovery.
- Do not add fuzzy matching as default edit behavior.
- Do not support arbitrary binary editing through these text tools.
- Do not include image, PDF, or notebook support in this roadmap; revisit those
  in a separate plan if rich-format workflows become a priority.
- Do not add compatibility shims for old file tool schemas unless a real
  external consumer requires it.
- Do not redact configured secret values from read or search output. This was
  considered and rejected: redaction risks corrupting legitimate file content
  (config samples, fixtures, docs) and degrading tool accuracy. Secret hygiene
  stays an operator concern enforced through `allowed_roots` scoping.
