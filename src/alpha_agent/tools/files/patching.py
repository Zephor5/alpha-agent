"""Patch mode parsing and application helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_agent.tools.files.errors import FileToolError
from alpha_agent.tools.files.io import read_text_file
from alpha_agent.tools.files.paths import ResolvedPath, resolve_write_path


@dataclass(frozen=True)
class RangeEdit:
    start_line: int
    end_line: int
    replacement: str


@dataclass(frozen=True)
class PlannedWrite:
    resolved: ResolvedPath
    before_text: str
    after_text: str
    before_sha256: str | None
    created: bool
    applied_edits: int


@dataclass(frozen=True)
class PatchOp:
    kind: str
    path: str
    lines: list[str]


def parse_range_edits(value: Any, *, line_count: int) -> list[RangeEdit]:
    if not isinstance(value, list) or not value:
        raise FileToolError("edits must be a non-empty array")
    edits: list[RangeEdit] = []
    previous_start = 0
    previous_covered_end = 0
    for raw_edit in value:
        if not isinstance(raw_edit, dict):
            raise FileToolError("each edit must be an object")
        start_line = _required_line_int(raw_edit.get("start_line"), "start_line", minimum=1)
        end_line = _required_line_int(raw_edit.get("end_line"), "end_line", minimum=0)
        replacement = raw_edit.get("replacement")
        if not isinstance(replacement, str):
            raise FileToolError("replacement must be a string")
        if "\x00" in replacement:
            raise FileToolError("binary files are not allowed")
        if end_line < start_line - 1:
            raise FileToolError("end_line must be at least start_line - 1")
        is_insertion = start_line == end_line + 1
        if is_insertion:
            if start_line > line_count + 1:
                raise FileToolError("insertion line must be at most line_count + 1")
            covered_end = end_line
        else:
            if end_line > line_count:
                raise FileToolError("end_line must be at most line_count")
            covered_end = end_line
        if start_line < previous_start:
            raise FileToolError("edits must be in original file order")
        if start_line == previous_start or start_line <= previous_covered_end:
            raise FileToolError("edits must not overlap")
        edits.append(RangeEdit(start_line, end_line, replacement))
        previous_start = start_line
        previous_covered_end = covered_end
    return edits


def apply_range_edits(before_text: str, edits: list[RangeEdit]) -> str:
    original_lines = before_text.splitlines(keepends=True)
    new_lines = list(original_lines)
    for edit in reversed(edits):
        new_lines[edit.start_line - 1 : edit.end_line] = edit.replacement.splitlines(keepends=True)
    return "".join(new_lines)


def apply_replace(
    before_text: str, old_string: str, new_string: str, *, replace_all: bool
) -> tuple[str, int]:
    if not old_string:
        raise FileToolError("old_string must be non-empty")
    if "\x00" in old_string or "\x00" in new_string:
        raise FileToolError("binary files are not allowed")
    count = before_text.count(old_string)
    if count == 0:
        raise FileToolError("old_string was not found")
    if count > 1 and not replace_all:
        raise FileToolError("old_string is not unique; set replace_all true to replace every match")
    return before_text.replace(old_string, new_string, -1 if replace_all else 1), count


def parse_patch_text(patch: str) -> list[PatchOp]:
    if "\x00" in patch:
        raise FileToolError("patch must not contain NUL characters")
    lines = patch.splitlines(keepends=True)
    stripped = [line.rstrip("\n") for line in lines]
    if stripped.count("*** Begin Patch") != 1 or stripped.count("*** End Patch") != 1:
        raise FileToolError("patch must contain exactly one Begin Patch and one End Patch")
    if not stripped or stripped[0] != "*** Begin Patch" or stripped[-1] != "*** End Patch":
        raise FileToolError("patch must start with Begin Patch and end with End Patch")
    ops: list[PatchOp] = []
    index = 1
    while index < len(lines) - 1:
        header = lines[index].rstrip("\n")
        if header.startswith("*** Add File: "):
            kind = "add"
            path = header.removeprefix("*** Add File: ").strip()
        elif header.startswith("*** Update File: "):
            kind = "update"
            path = header.removeprefix("*** Update File: ").strip()
        elif header.startswith("*** Delete File") or header.startswith("*** Move"):
            raise FileToolError("delete and move patch operations are not supported")
        else:
            raise FileToolError("patch operation header is malformed")
        if not path:
            raise FileToolError("patch path must be non-empty")
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not lines[index].startswith("*** "):
            body.append(lines[index])
            index += 1
        ops.append(PatchOp(kind, path, body))
    if not ops:
        raise FileToolError("patch must contain at least one operation")
    return ops


def plan_patch_text(
    patch: str,
    *,
    write_roots: tuple[Path, ...],
    max_file_bytes: int,
) -> list[PlannedWrite]:
    ops = parse_patch_text(patch)
    planned: list[PlannedWrite] = []
    seen: set[Path] = set()
    for op in ops:
        resolved = resolve_write_path(
            op.path,
            roots=write_roots,
            create_if_missing=op.kind == "add",
            reject_absolute=True,
        )
        key = resolved.path.resolve(strict=False)
        if key in seen:
            raise FileToolError("patch must not contain duplicate paths")
        seen.add(key)
        if op.kind == "add":
            if resolved.path.exists():
                raise FileToolError("Add File target already exists")
            after_text = _parse_add_body(op.lines)
            planned.append(
                PlannedWrite(
                    resolved=resolved,
                    before_text="",
                    after_text=after_text,
                    before_sha256=None,
                    created=True,
                    applied_edits=1,
                )
            )
            continue
        if not resolved.path.exists():
            raise FileToolError("Update File target does not exist")
        data, before_text = read_text_file(resolved.path, max_file_bytes=max_file_bytes)
        after_text, applied = _apply_update_body(before_text, op.lines)
        planned.append(
            PlannedWrite(
                resolved=resolved,
                before_text=before_text,
                after_text=after_text,
                before_sha256=hashlib.sha256(data).hexdigest(),
                created=False,
                applied_edits=applied,
            )
        )
    return planned


def resolve_patch_text_paths(patch: str, *, write_roots: tuple[Path, ...]) -> list[Path]:
    ops = parse_patch_text(patch)
    paths: list[Path] = []
    seen: set[Path] = set()
    for op in ops:
        resolved = resolve_write_path(
            op.path,
            roots=write_roots,
            create_if_missing=op.kind == "add",
            reject_absolute=True,
        )
        key = resolved.path.resolve(strict=False)
        if key in seen:
            raise FileToolError("patch must not contain duplicate paths")
        seen.add(key)
        paths.append(resolved.path)
    return paths


def _parse_add_body(lines: list[str]) -> str:
    content: list[str] = []
    for line in lines:
        if not line.startswith("+"):
            raise FileToolError("Add File lines must start with +")
        content.append(line[1:])
    return "".join(content)


def _apply_update_body(before_text: str, lines: list[str]) -> tuple[str, int]:
    current_lines = before_text.splitlines(keepends=True)
    index = 0
    applied = 0
    while index < len(lines):
        if lines[index].startswith("@@"):
            index += 1
            continue
        before_seq: list[str] = []
        after_seq: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            if line.startswith(" "):
                raw = line[1:]
                before_seq.append(raw)
                after_seq.append(raw)
            elif line.startswith("-"):
                before_seq.append(line[1:])
            elif line.startswith("+"):
                after_seq.append(line[1:])
            else:
                raise FileToolError("Update File hunk lines must start with space, -, +, or @@")
            index += 1
        if not before_seq:
            raise FileToolError("Update File hunks require context or removed lines")
        match_index = _unique_subsequence_index(current_lines, before_seq)
        current_lines[match_index : match_index + len(before_seq)] = after_seq
        applied += 1
    if applied == 0:
        raise FileToolError("Update File must contain at least one hunk")
    return "".join(current_lines), applied


def _unique_subsequence_index(lines: list[str], needle: list[str]) -> int:
    matches: list[int] = []
    if len(needle) > len(lines):
        raise FileToolError("patch context does not match target file")
    for index in range(0, len(lines) - len(needle) + 1):
        if lines[index : index + len(needle)] == needle:
            matches.append(index)
            if len(matches) > 1:
                raise FileToolError("patch context is ambiguous")
    if not matches:
        raise FileToolError("patch context does not match target file")
    return matches[0]


def _required_line_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < minimum:
        raise FileToolError(f"{field_name} must be at least {minimum}")
    return value
