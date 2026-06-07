"""Ripgrep-backed file discovery and content search."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_agent.tools.files.errors import FileToolError
from alpha_agent.tools.files.paths import DEFAULT_EXCLUDED_DIRS

MAX_SEARCH_LINE_CHARS = 500
DisplayFunc = Callable[[Path], str]


def rg_available() -> bool:
    return shutil.which("rg") is not None


def rg_files(
    root: Path,
    *,
    pattern: str,
    max_depth: int | None,
) -> list[Path]:
    if not rg_available():
        raise FileToolError("rg is unavailable; recursive file discovery requires ripgrep")
    cmd = ["rg", "--files", "--color", "never"]
    if max_depth is not None:
        cmd.extend(["--max-depth", str(max_depth)])
    for dirname in sorted(DEFAULT_EXCLUDED_DIRS):
        cmd.extend(["-g", f"!{dirname}/**"])
    if pattern and pattern != "*":
        cmd.extend(["-g", pattern])
    completed = subprocess.run(
        cmd,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode not in (0, 1):
        raise FileToolError(f"rg --files failed: {completed.stderr.strip()}")
    return [root / line for line in completed.stdout.splitlines() if line]


def rg_search(
    path: Path,
    *,
    pattern: str,
    mode: str,
    glob: str | None,
    file_type: str | None,
    output_mode: str,
    case_sensitive: bool,
    before_context: int,
    after_context: int,
    limit: int,
    offset: int,
    multiline: bool,
    max_file_bytes: int,
    display: DisplayFunc,
) -> dict[str, Any]:
    if not rg_available():
        payload: dict[str, Any] = {
            "error": "rg is unavailable; file_search requires ripgrep",
            "limit": limit,
            "offset": offset,
            "next_offset": None,
            "truncated": False,
        }
        if output_mode == "content":
            payload.update({"matches": [], "match_count": 0})
        elif output_mode == "files_with_matches":
            payload.update({"files": [], "file_count": 0})
        elif output_mode == "count":
            payload.update({"counts": [], "total_matches": 0, "file_count": 0})
        return payload
    if output_mode == "files_with_matches":
        return _search_files(
            path,
            pattern,
            mode,
            glob,
            file_type,
            case_sensitive,
            limit,
            offset,
            multiline,
            max_file_bytes,
            display,
        )
    events = _run_rg_json(
        path,
        pattern,
        mode,
        glob,
        file_type,
        case_sensitive,
        before_context,
        after_context,
        multiline,
        max_file_bytes,
    )
    if output_mode == "count":
        return _count_output(events, limit=limit, offset=offset, display=display)
    return _content_output(
        events,
        limit=limit,
        offset=offset,
        before_context=before_context,
        after_context=after_context,
        display=display,
    )


def _base_rg_cmd(
    pattern: str,
    mode: str,
    glob: str | None,
    file_type: str | None,
    case_sensitive: bool,
    multiline: bool,
    max_file_bytes: int,
) -> list[str]:
    cmd = ["rg", "--color", "never", "--no-heading", "--max-filesize", str(max_file_bytes)]
    if mode == "literal":
        cmd.append("-F")
    elif mode != "regex":
        raise FileToolError("mode must be one of: regex, literal")
    if not case_sensitive:
        cmd.append("-i")
    else:
        cmd.append("-s")
    if multiline:
        cmd.extend(["-U", "--multiline"])
    if glob:
        cmd.extend(["-g", glob])
    if file_type:
        cmd.extend(["-t", file_type])
    for dirname in sorted(DEFAULT_EXCLUDED_DIRS):
        cmd.extend(["-g", f"!{dirname}/**"])
    return cmd


def _run_rg_json(
    path: Path,
    pattern: str,
    mode: str,
    glob: str | None,
    file_type: str | None,
    case_sensitive: bool,
    before_context: int,
    after_context: int,
    multiline: bool,
    max_file_bytes: int,
) -> list[dict[str, Any]]:
    cmd = _base_rg_cmd(pattern, mode, glob, file_type, case_sensitive, multiline, max_file_bytes)
    cmd.extend(["--json"])
    if before_context:
        cmd.extend(["--before-context", str(before_context)])
    if after_context:
        cmd.extend(["--after-context", str(after_context)])
    cmd.extend(["--", pattern, str(path)])
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    if completed.returncode not in (0, 1):
        raise FileToolError(f"rg search failed: {completed.stderr.strip()}")
    events: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _search_files(
    path: Path,
    pattern: str,
    mode: str,
    glob: str | None,
    file_type: str | None,
    case_sensitive: bool,
    limit: int,
    offset: int,
    multiline: bool,
    max_file_bytes: int,
    display: DisplayFunc,
) -> dict[str, Any]:
    cmd = _base_rg_cmd(pattern, mode, glob, file_type, case_sensitive, multiline, max_file_bytes)
    cmd.append("-l")
    cmd.extend(["--", pattern, str(path)])
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    if completed.returncode not in (0, 1):
        raise FileToolError(f"rg search failed: {completed.stderr.strip()}")
    all_paths = [Path(line) for line in completed.stdout.splitlines() if line]
    all_paths.sort(key=lambda item: (-item.stat().st_mtime, display(item)))
    selected = all_paths[offset : offset + limit]
    files = [
        {
            "path": display(item),
            "size": item.stat().st_size,
            "mtime": item.stat().st_mtime,
        }
        for item in selected
    ]
    next_offset = offset + len(selected) if offset + len(selected) < len(all_paths) else None
    return {
        "files": files,
        "file_count": len(all_paths),
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "truncated": next_offset is not None,
    }


def _content_output(
    events: list[dict[str, Any]],
    *,
    limit: int,
    offset: int,
    before_context: int,
    after_context: int,
    display: DisplayFunc,
) -> dict[str, Any]:
    context_events: list[tuple[str, int, dict[str, Any]]] = []
    matches: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        data = event.get("data", {})
        path_text = data.get("path", {}).get("text")
        line_number = data.get("line_number")
        if event_type == "context" and path_text and isinstance(line_number, int):
            line = _truncate_line(data.get("lines", {}).get("text", "").rstrip("\n"))
            context_events.append(
                (path_text, line_number, {"line_number": line_number, "line": line})
            )
        if event_type != "match" or not path_text or not isinstance(line_number, int):
            continue
        line = _truncate_line(data.get("lines", {}).get("text", "").rstrip("\n"))
        matches.append(
            {
                "path": display(Path(path_text)),
                "_raw_path": path_text,
                "line_number": line_number,
                "line": line,
                "before": [],
                "after": [],
            }
        )
    for match in matches:
        match_path = match["_raw_path"]
        match_line = match["line_number"]
        if not isinstance(match_path, str) or not isinstance(match_line, int):
            continue
        before: list[dict[str, Any]] = []
        after: list[dict[str, Any]] = []
        for context_path, context_line, payload in context_events:
            if context_path != match_path:
                continue
            before_distance = match_line - context_line
            after_distance = context_line - match_line
            if 1 <= before_distance <= before_context:
                before.append(payload)
            elif 1 <= after_distance <= after_context:
                after.append(payload)
        match["before"] = sorted(before, key=lambda item: item["line_number"])
        match["after"] = sorted(after, key=lambda item: item["line_number"])
        del match["_raw_path"]
    selected = matches[offset : offset + limit]
    next_offset = offset + len(selected) if offset + len(selected) < len(matches) else None
    return {
        "matches": selected,
        "match_count": len(matches),
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "truncated": next_offset is not None,
    }


def _count_output(
    events: list[dict[str, Any]],
    *,
    limit: int,
    offset: int,
    display: DisplayFunc,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path_text = data.get("path", {}).get("text")
        if path_text:
            counts[path_text] = counts.get(path_text, 0) + 1
    rows = [
        {"path": display(Path(path_text)), "count": count}
        for path_text, count in sorted(counts.items(), key=lambda item: item[0])
    ]
    selected = rows[offset : offset + limit]
    next_offset = offset + len(selected) if offset + len(selected) < len(rows) else None
    return {
        "counts": selected,
        "total_matches": sum(counts.values()),
        "file_count": len(counts),
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "truncated": next_offset is not None,
    }


def _truncate_line(line: str) -> str:
    if len(line) <= MAX_SEARCH_LINE_CHARS:
        return line
    return line[:MAX_SEARCH_LINE_CHARS]
