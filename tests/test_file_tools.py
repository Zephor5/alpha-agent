from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from alpha_agent.config import AlphaConfig, FileToolConfig
from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.files import (
    FILE_GLOB_TOOL_NAME,
    FILE_PATCH_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_SEARCH_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    FileGlobTool,
    FilePatchTool,
    FileReadTool,
    FileSearchTool,
    FileToolError,
    FileWriteTool,
)
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME


def _tool_context(tmp_path: Path, *, turn_state: object | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path,
        check_canceled=lambda _stage: None,
        extensions={"turn_state": turn_state} if turn_state is not None else {},
    )


def _file_config(root: Path, **overrides: object) -> FileToolConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "allowed_roots": (root.resolve(),),
        "patch_enabled": False,
        "write_roots": (),
        "max_read_chars": 20000,
        "max_file_bytes": 1000000,
        "max_search_results": 100,
        "max_glob_results": 500,
        "max_read_lines": 200,
        "create_parent_dirs_enabled": False,
        "max_output_chars": 30000,
    }
    values.update(overrides)
    return FileToolConfig(**values)


def test_default_registry_uses_target_file_toolset(tmp_path: Path) -> None:
    default_enabled = AlphaConfig(
        db_path=tmp_path / "enabled.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
    )
    patch_enabled = AlphaConfig(
        db_path=tmp_path / "patch.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        file_tool=FileToolConfig(
            enabled=True,
            allowed_roots=(tmp_path,),
            patch_enabled=True,
            write_roots=(tmp_path,),
        ),
    )

    assert build_tool_registry(default_enabled).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_GLOB_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
    ]
    assert build_tool_registry(patch_enabled).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_GLOB_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
        FILE_PATCH_TOOL_NAME,
        FILE_WRITE_TOOL_NAME,
    ]


def test_file_glob_single_directory_browse_includes_dirs_and_paginates(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "secret.txt").write_text("hidden\n", encoding="utf-8")

    result = FileGlobTool(_file_config(root, max_glob_results=1)).run(
        {"path": ".", "max_depth": 1, "include_dirs": True, "limit": 99, "sort": "path_asc"},
        _tool_context(tmp_path),
    )

    output = cast(dict[str, object], result.output)
    files = cast(list[dict[str, object]], output["files"])
    assert output["total_count"] == 2
    assert output["truncated"] is True
    assert output["next_offset"] == 1
    assert files == [
        {
            "path": "a.txt",
            "type": "file",
            "size": 2,
            "mtime": files[0]["mtime"],
        }
    ]


def test_file_glob_recursive_requires_rg_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("print('a')\n", encoding="utf-8")
    monkeypatch.setattr("alpha_agent.tools.files.searching.shutil.which", lambda _name: None)

    result = FileGlobTool(_file_config(root)).run(
        {"path": ".", "pattern": "*.py"},
        _tool_context(tmp_path),
    )

    output = cast(dict[str, object], result.output)
    assert output["files"] == []
    assert "requires ripgrep" in str(output["error"])
    assert result.metadata["unavailable"] is True


def test_file_glob_recursive_include_dirs_prunes_excluded_subtrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "src" / "pkg").mkdir(parents=True)
    (root / ".git" / "objects").mkdir(parents=True)
    (root / "node_modules" / "pkg").mkdir(parents=True)
    monkeypatch.setattr("alpha_agent.tools.files.tools.rg_files", lambda *_args, **_kwargs: [])

    result = FileGlobTool(_file_config(root)).run(
        {
            "path": ".",
            "pattern": "*",
            "max_depth": 3,
            "include_dirs": True,
            "sort": "path_asc",
        },
        _tool_context(tmp_path),
    )

    output = cast(dict[str, object], result.output)
    files = cast(list[dict[str, object]], output["files"])
    paths = [str(item["path"]) for item in files]
    assert "src" in paths
    assert "src/pkg" in paths
    assert ".git/objects" not in paths
    assert "node_modules/pkg" not in paths


def test_file_read_returns_target_contract_and_deduplicates_turn_reads(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    text = "line1\nline2\nline3\n"
    (root / "sample.txt").write_text(text, encoding="utf-8")
    state = SimpleNamespace()
    context = _tool_context(tmp_path, turn_state=state)
    tool = FileReadTool(_file_config(root))

    first = tool.run({"path": "sample.txt", "offset": 2, "limit": 1}, context)
    second = tool.run({"path": "sample.txt", "offset": 2, "limit": 1}, context)

    output = cast(dict[str, object], first.output)
    assert output["content"] == "     2\tline2\n"
    assert output["returned_lines"] == 1
    assert output["total_lines"] == 3
    assert output["next_offset"] == 3
    assert output["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert output["size"] == len(text.encode("utf-8"))
    second_output = cast(dict[str, object], second.output)
    assert second_output["deduplicated"] is True
    assert second_output["content"] == "<same file view already read this turn>"


def test_file_read_rejects_binary_and_suggests_missing_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.txt").write_text("text\n", encoding="utf-8")
    (root / "binary.dat").write_bytes(b"text\x00binary")
    tool = FileReadTool(_file_config(root))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run({"path": "binary.dat"}, _tool_context(tmp_path))
    with pytest.raises(FileToolError, match="suggestions: sample.txt"):
        tool.run({"path": "sampel.txt"}, _tool_context(tmp_path))


def test_file_read_suggests_likely_project_relative_missing_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "foo.py").write_text("src\n", encoding="utf-8")
    (root / "tests" / "foo.py").write_text("tests\n", encoding="utf-8")
    tool = FileReadTool(_file_config(root))

    with pytest.raises(FileToolError) as exc_info:
        tool.run({"path": "foo.py"}, _tool_context(tmp_path))

    message = str(exc_info.value)
    assert "suggestions:" in message
    assert "src/foo.py" in message
    assert "tests/foo.py" in message


def test_file_search_literal_content_context_and_count_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    a_path = root / "a.txt"
    b_path = root / "b.txt"
    a_path.write_text("before\nneedle.one\nafter\nneedle.two\n", encoding="utf-8")
    b_path.write_text("needle.one\n", encoding="utf-8")
    tool = FileSearchTool(_file_config(root, max_search_results=2))

    def rg_event(event_type: str, path: Path, line_number: int, line: str) -> str:
        return json.dumps(
            {
                "type": event_type,
                "data": {
                    "path": {"text": str(path)},
                    "line_number": line_number,
                    "lines": {"text": f"{line}\n"},
                },
            }
        )

    def fake_rg_run(_cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        pattern = _cmd[_cmd.index("--") + 1]
        if pattern == "needle.one":
            stdout = "\n".join(
                [
                    rg_event("context", a_path, 1, "before"),
                    rg_event("match", a_path, 2, "needle.one"),
                    rg_event("context", a_path, 3, "after"),
                    rg_event("match", b_path, 1, "needle.one"),
                ]
            )
        else:
            stdout = "\n".join(
                [
                    rg_event("match", a_path, 2, "needle.one"),
                    rg_event("match", a_path, 4, "needle.two"),
                    rg_event("match", b_path, 1, "needle.one"),
                ]
            )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("alpha_agent.tools.files.searching.shutil.which", lambda _name: "rg")
    monkeypatch.setattr("alpha_agent.tools.files.searching.subprocess.run", fake_rg_run)

    content = tool.run(
        {
            "pattern": "needle.one",
            "mode": "literal",
            "path": ".",
            "output_mode": "content",
            "context": 1,
            "limit": 10,
            "sort": "path_asc",
        },
        _tool_context(tmp_path),
    )
    count = tool.run(
        {"pattern": "needle", "path": ".", "output_mode": "count"},
        _tool_context(tmp_path),
    )

    content_output = cast(dict[str, object], content.output)
    matches = cast(list[dict[str, object]], content_output["matches"])
    assert len(matches) == 2
    assert matches[0]["line"] == "needle.one"
    assert matches[0]["before"] == [{"line_number": 1, "line": "before"}]
    assert matches[0]["after"] == [{"line_number": 3, "line": "after"}]
    count_output = cast(dict[str, object], count.output)
    assert count_output["total_matches"] == 3
    assert count_output["file_count"] == 2


def test_file_search_rg_unavailable_returns_deterministic_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr("alpha_agent.tools.files.searching.shutil.which", lambda _name: None)

    result = FileSearchTool(_file_config(root)).run(
        {"pattern": "needle", "path": ".", "output_mode": "content", "limit": 3, "offset": 2},
        _tool_context(tmp_path),
    )

    output = cast(dict[str, object], result.output)
    assert output["error"] == "rg is unavailable; file_search requires ripgrep"
    assert output["matches"] == []
    assert output["limit"] == 3
    assert output["offset"] == 2
    assert output["next_offset"] is None
    assert output["truncated"] is False
    assert "files" not in output
    assert "counts" not in output
    assert result.metadata["unavailable"] is True


def test_file_search_rg_unavailable_shapes_follow_output_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr("alpha_agent.tools.files.searching.shutil.which", lambda _name: None)
    tool = FileSearchTool(_file_config(root))

    files = tool.run(
        {"pattern": "needle", "path": ".", "output_mode": "files_with_matches"},
        _tool_context(tmp_path),
    )
    counts = tool.run(
        {"pattern": "needle", "path": ".", "output_mode": "count"},
        _tool_context(tmp_path),
    )

    files_output = cast(dict[str, object], files.output)
    assert files_output["files"] == []
    assert files_output["file_count"] == 0
    assert "matches" not in files_output
    assert "counts" not in files_output
    counts_output = cast(dict[str, object], counts.output)
    assert counts_output["counts"] == []
    assert counts_output["total_matches"] == 0
    assert counts_output["file_count"] == 0
    assert "matches" not in counts_output
    assert "files" not in counts_output


def test_file_patch_replace_unique_and_replace_all(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "sample.txt"
    original = "one\ntwo\ntwo\n"
    path.write_text(original, encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="not unique"):
        tool.run(
            {
                "mode": "replace",
                "path": "sample.txt",
                "expected_sha256": _sha256_text(original),
                "old_string": "two",
                "new_string": "TWO",
            },
            _tool_context(tmp_path),
        )

    result = tool.run(
        {
            "mode": "replace",
            "path": "sample.txt",
            "expected_sha256": _sha256_text(original),
            "old_string": "two",
            "new_string": "TWO",
            "replace_all": True,
        },
        _tool_context(tmp_path),
    )

    assert path.read_text(encoding="utf-8") == "one\nTWO\nTWO\n"
    output = cast(dict[str, object], result.output)
    assert output["applied_edits"] == 2
    assert output["files_modified"] == ["sample.txt"]


def test_file_patch_patch_text_add_update_and_no_partial_context_failure(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    existing = root / "existing.txt"
    existing.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    result = tool.run(
        {
            "mode": "patch_text",
            "patch": """*** Begin Patch
*** Update File: existing.txt
@@
 alpha
-beta
+BETA
 gamma
*** Add File: created.txt
+new
*** End Patch""",
        },
        _tool_context(tmp_path),
    )

    assert existing.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert (root / "created.txt").read_text(encoding="utf-8") == "new\n"
    output = cast(dict[str, object], result.output)
    assert output["files_modified"] == ["existing.txt"]
    assert output["files_created"] == ["created.txt"]

    before = existing.read_text(encoding="utf-8")
    with pytest.raises(FileToolError, match="context does not match"):
        tool.run(
            {
                "mode": "patch_text",
                "patch": """*** Begin Patch
*** Update File: existing.txt
@@
 alpha
-missing
+MISS
 gamma
*** Add File: should_not_exist.txt
+nope
*** End Patch""",
            },
            _tool_context(tmp_path),
        )
    assert existing.read_text(encoding="utf-8") == before
    assert not (root / "should_not_exist.txt").exists()


def test_file_patch_patch_text_prevents_partial_writes_for_oversized_later_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")
    tool = FilePatchTool(
        _file_config(root, patch_enabled=True, write_roots=(root.resolve(),), max_file_bytes=12)
    )

    with pytest.raises(FileToolError, match="file is too large to write"):
        tool.run(
            {
                "mode": "patch_text",
                "patch": """*** Begin Patch
*** Update File: first.txt
@@
-alpha
+ALPHA
*** Update File: second.txt
@@
-beta
+this is too large
*** End Patch""",
            },
            _tool_context(tmp_path),
        )

    assert first.read_text(encoding="utf-8") == "alpha\n"
    assert second.read_text(encoding="utf-8") == "beta\n"


def test_file_patch_range_rechecks_expected_sha256_inside_path_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "sample.txt"
    original = "one\n"
    changed = "changed\n"
    path.write_text(original, encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))
    rewrite_once = {"done": False}
    real_path_locks = __import__(
        "alpha_agent.tools.files.tools", fromlist=["path_locks"]
    ).path_locks

    def mutate_before_lock(paths: list[Path]) -> object:
        if not rewrite_once["done"]:
            rewrite_once["done"] = True
            path.write_text(changed, encoding="utf-8")
        return real_path_locks(paths)

    monkeypatch.setattr("alpha_agent.tools.files.tools.path_locks", mutate_before_lock)

    with pytest.raises(FileToolError, match="expected_sha256 does not match"):
        tool.run(
            {
                "mode": "range",
                "path": "sample.txt",
                "expected_sha256": _sha256_text(original),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "two\n"}],
            },
            _tool_context(tmp_path),
        )

    assert path.read_text(encoding="utf-8") == changed


def test_file_patch_rejects_absolute_patch_text_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="absolute paths"):
        tool.run(
            {
                "mode": "patch_text",
                "patch": f"""*** Begin Patch
*** Add File: {root / "abs.txt"}
+no
*** End Patch""",
            },
            _tool_context(tmp_path),
        )


def test_file_write_create_overwrite_hash_and_parent_gate(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    tool = FileWriteTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    created = tool.run({"path": "sample.txt", "content": "one\n"}, _tool_context(tmp_path))
    assert (root / "sample.txt").read_text(encoding="utf-8") == "one\n"
    assert cast(dict[str, object], created.output)["files_created"] == ["sample.txt"]

    with pytest.raises(FileToolError, match="overwrite must be true"):
        tool.run({"path": "sample.txt", "content": "two\n"}, _tool_context(tmp_path))
    with pytest.raises(FileToolError, match="does not match"):
        tool.run(
            {
                "path": "sample.txt",
                "content": "two\n",
                "overwrite": True,
                "expected_sha256": _sha256_text("wrong\n"),
            },
            _tool_context(tmp_path),
        )
    with pytest.raises(FileToolError, match="create_parent_dirs is disabled"):
        tool.run(
            {"path": "nested/new.txt", "content": "new\n", "create_parent_dirs": True},
            _tool_context(tmp_path),
        )
    parent_enabled = FileWriteTool(
        _file_config(
            root,
            patch_enabled=True,
            write_roots=(root.resolve(),),
            create_parent_dirs_enabled=True,
        )
    )
    with pytest.raises(FileToolError, match="outside tools.files.write_roots"):
        parent_enabled.run(
            {"path": "../outside/new.txt", "content": "new\n", "create_parent_dirs": True},
            _tool_context(tmp_path),
        )
    assert not (tmp_path / "outside").exists()

    result = tool.run(
        {
            "path": "sample.txt",
            "content": "two\n",
            "overwrite": True,
            "expected_sha256": _sha256_text("one\n"),
        },
        _tool_context(tmp_path),
    )
    output = cast(dict[str, object], result.output)
    assert output["files_modified"] == ["sample.txt"]
    assert "-one" in str(output["diff"])
    assert "+two" in str(output["diff"])


def test_file_write_rechecks_expected_sha256_inside_path_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "sample.txt"
    original = "one\n"
    changed = "changed\n"
    path.write_text(original, encoding="utf-8")
    tool = FileWriteTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))
    rewrite_once = {"done": False}
    real_path_locks = __import__(
        "alpha_agent.tools.files.tools", fromlist=["path_locks"]
    ).path_locks

    def mutate_before_lock(paths: list[Path]) -> object:
        if not rewrite_once["done"]:
            rewrite_once["done"] = True
            path.write_text(changed, encoding="utf-8")
        return real_path_locks(paths)

    monkeypatch.setattr("alpha_agent.tools.files.tools.path_locks", mutate_before_lock)

    with pytest.raises(FileToolError, match="expected_sha256 does not match"):
        tool.run(
            {
                "path": "sample.txt",
                "content": "two\n",
                "overwrite": True,
                "expected_sha256": _sha256_text(original),
            },
            _tool_context(tmp_path),
        )

    assert path.read_text(encoding="utf-8") == changed


def test_file_write_invalidates_read_ledger_when_atomic_write_fails_after_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "sample.txt"
    original = "one\n"
    replacement = "two\n"
    fixed_mtime_ns = 1_700_000_000_000_000_000
    path.write_text(original, encoding="utf-8")
    os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    state = SimpleNamespace()
    context = _tool_context(tmp_path, turn_state=state)
    read_tool = FileReadTool(_file_config(root))
    write_tool = FileWriteTool(
        _file_config(root, patch_enabled=True, write_roots=(root.resolve(),))
    )

    first = read_tool.run({"path": "sample.txt", "offset": 1, "limit": 1}, context)
    assert cast(dict[str, object], first.output)["content"] == "     1\tone\n"

    def fail_after_mutation(
        target: Path,
        content: str,
        *,
        max_file_bytes: int,
    ) -> tuple[int, str]:
        del max_file_bytes
        assert target == path
        path.write_text(content, encoding="utf-8")
        os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
        raise FileToolError("post-write verification failed")

    monkeypatch.setattr("alpha_agent.tools.files.tools.atomic_write_text", fail_after_mutation)

    with pytest.raises(FileToolError, match="post-write verification failed"):
        write_tool.run(
            {
                "path": "sample.txt",
                "content": replacement,
                "overwrite": True,
                "expected_sha256": _sha256_text(original),
            },
            context,
        )

    second = read_tool.run({"path": "sample.txt", "offset": 1, "limit": 1}, context)
    second_output = cast(dict[str, object], second.output)
    assert "deduplicated" not in second_output
    assert second_output["content"] == "     1\ttwo\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
