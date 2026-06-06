from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from alpha_agent.config import AlphaConfig, FileToolConfig
from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.files import (
    FILE_LIST_TOOL_NAME,
    FILE_PATCH_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_SEARCH_TOOL_NAME,
    FileListTool,
    FilePatchTool,
    FileReadTool,
    FileSearchTool,
    FileToolError,
)
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME


def _tool_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path,
        check_canceled=lambda _stage: None,
    )


def _file_config(root: Path, **overrides: object) -> FileToolConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "allowed_roots": (root.resolve(),),
        "patch_enabled": False,
        "write_roots": (),
        "max_read_chars": 20000,
        "max_file_bytes": 1000000,
        "max_search_matches": 100,
        "max_list_entries": 500,
        "max_output_chars": 30000,
    }
    values.update(overrides)
    return FileToolConfig(**values)


def test_file_tools_declare_read_only_governance(tmp_path: Path) -> None:
    config = _file_config(tmp_path, max_read_chars=123, max_list_entries=7)

    list_spec = FileListTool(config).spec
    read_spec = FileReadTool(config).spec
    search_spec = FileSearchTool(config).spec

    assert list_spec.toolset == "file"
    assert list_spec.read_only is True
    assert list_spec.destructive is False
    assert list_spec.concurrency_safe is True
    assert "group" not in list_spec.to_dict()
    assert list_spec.parameters["properties"]["max_entries"]["maximum"] == 7
    assert read_spec.parameters["properties"]["max_chars"]["maximum"] == 123
    assert search_spec.toolset == "file"


def test_file_patch_declares_write_governance(tmp_path: Path) -> None:
    config = _file_config(tmp_path, patch_enabled=True, write_roots=(tmp_path.resolve(),))

    patch_spec = FilePatchTool(config).spec

    assert patch_spec.name == FILE_PATCH_TOOL_NAME
    assert patch_spec.toolset == "file"
    assert patch_spec.read_only is False
    assert patch_spec.destructive is True
    assert patch_spec.concurrency_safe is False
    assert patch_spec.requires_user_interaction is False
    assert "group" not in patch_spec.to_dict()


def test_file_list_returns_bounded_entries_and_skips_default_excluded_dirs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("beta\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "secret.txt").write_text("hidden\n", encoding="utf-8")

    tool = FileListTool(_file_config(root, max_list_entries=2))

    result = tool.run(
        {"path": ".", "recursive": True, "max_entries": 99},
        _tool_context(tmp_path),
    )

    assert isinstance(result.output, dict)
    output = cast(dict[str, object], result.output)
    entries = output["entries"]
    assert isinstance(entries, list)
    assert len(entries) == 2
    assert output["truncated"] is True
    paths = {entry["path"] for entry in entries if isinstance(entry, dict)}
    assert ".git/secret.txt" not in paths
    assert paths <= {"a.txt", "nested", "nested/b.txt"}
    for entry in entries:
        assert isinstance(entry, dict)
        assert {"path", "type", "size", "mtime", "truncated"} <= set(entry)


def test_file_read_returns_line_range_hash_and_truncation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    text = "line1\nline2\nline3\nline4\n"
    path = root / "sample.txt"
    path.write_text(text, encoding="utf-8")
    tool = FileReadTool(_file_config(root))

    result = tool.run(
        {"path": "sample.txt", "start_line": 2, "end_line": 3, "max_chars": 100},
        _tool_context(tmp_path),
    )

    assert isinstance(result.output, dict)
    output = cast(dict[str, object], result.output)
    assert output == {
        "path": "sample.txt",
        "content": "line2\nline3\n",
        "start_line": 2,
        "end_line": 3,
        "truncated": True,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "size": len(text.encode("utf-8")),
    }

    truncated = tool.run(
        {"path": "sample.txt", "max_chars": 5},
        _tool_context(tmp_path),
    )
    assert isinstance(truncated.output, dict)
    truncated_output = cast(dict[str, object], truncated.output)
    assert truncated_output["content"] == "line1"
    assert truncated_output["end_line"] == 1
    assert truncated_output["truncated"] is True


def test_file_read_reports_actual_end_line_when_start_is_beyond_eof(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.txt").write_text("line1\nline2\n", encoding="utf-8")
    tool = FileReadTool(_file_config(root))

    result = tool.run(
        {"path": "sample.txt", "start_line": 99},
        _tool_context(tmp_path),
    )

    assert isinstance(result.output, dict)
    output = cast(dict[str, object], result.output)
    assert output["content"] == ""
    assert output["start_line"] == 99
    assert output["end_line"] == 2


def test_file_read_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    tool = FileReadTool(_file_config(root))

    with pytest.raises(FileToolError, match="outside tools.files.allowed_roots"):
        tool.run({"path": str(outside)}, _tool_context(tmp_path))


def test_file_read_rejects_binary_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(b"text\x00binary")
    tool = FileReadTool(_file_config(root))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run({"path": "binary.dat"}, _tool_context(tmp_path))


def test_file_search_returns_context_and_applies_match_limit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text(
        "before\nneedle one\nafter\nneedle two\n",
        encoding="utf-8",
    )
    (root / "b.md").write_text("needle ignored by glob\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "secret.txt").write_text("needle hidden\n", encoding="utf-8")
    tool = FileSearchTool(_file_config(root, max_search_matches=1))

    result = tool.run(
        {
            "query": "NEEDLE",
            "path": ".",
            "glob": "*.txt",
            "context_lines": 1,
            "max_matches": 99,
        },
        _tool_context(tmp_path),
    )

    assert isinstance(result.output, dict)
    output = cast(dict[str, object], result.output)
    matches = output["matches"]
    assert isinstance(matches, list)
    assert len(matches) == 1
    assert output["truncated"] is True
    match = matches[0]
    assert isinstance(match, dict)
    assert match["path"] == "a.txt"
    assert match["line_number"] == 2
    assert match["line"] == "needle one"
    assert match["context"] == {
        "before": [{"line_number": 1, "line": "before"}],
        "after": [{"line_number": 3, "line": "after"}],
    }


def test_file_search_rejects_explicit_binary_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(b"needle\x00binary")
    tool = FileSearchTool(_file_config(root))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run({"query": "needle", "path": "binary.dat"}, _tool_context(tmp_path))


def test_file_tools_do_not_descend_into_symlink_directories(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (real / "hidden.txt").write_text("needle hidden\n", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    list_result = FileListTool(_file_config(root)).run(
        {"path": ".", "recursive": True},
        _tool_context(tmp_path),
    )
    search_result = FileSearchTool(_file_config(root)).run(
        {"query": "needle", "path": "linked"},
        _tool_context(tmp_path),
    )

    assert isinstance(list_result.output, dict)
    entries = cast(list[dict[str, object]], list_result.output["entries"])
    entry_by_path = {str(entry["path"]): entry for entry in entries}
    assert entry_by_path["linked"]["type"] == "symlink"
    assert "linked/hidden.txt" not in entry_by_path
    assert isinstance(search_result.output, dict)
    assert search_result.output["matches"] == []


def test_file_patch_replaces_and_inserts_with_expected_hash(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = "one\ntwo\nthree\n"
    path = root / "sample.txt"
    path.write_text(original, encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    result = tool.run(
        {
            "path": "sample.txt",
            "expected_sha256": _sha256_text(original),
            "edits": [
                {"start_line": 2, "end_line": 2, "replacement": "TWO\n"},
                {"start_line": 4, "end_line": 3, "replacement": "four\n"},
            ],
        },
        _tool_context(tmp_path),
    )

    assert path.read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\n"
    assert isinstance(result.output, dict)
    output = cast(dict[str, object], result.output)
    assert output["path"] == "sample.txt"
    assert output["before_sha256"] == _sha256_text(original)
    assert output["after_sha256"] == _sha256_text("one\nTWO\nthree\nfour\n")
    assert output["bytes_written"] == len(b"one\nTWO\nthree\nfour\n")
    assert output["line_count"] == 4
    assert output["applied_edits"] == 2
    assert isinstance(output["diff"], str)
    assert "-two" in output["diff"]
    assert "+TWO" in output["diff"]


def test_file_patch_creates_file_only_when_explicit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="create_if_missing must be true"):
        tool.run(
            {
                "path": "created.txt",
                "edits": [{"start_line": 1, "end_line": 0, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )

    result = tool.run(
        {
            "path": "created.txt",
            "expected_sha256": "",
            "create_if_missing": True,
            "edits": [{"start_line": 1, "end_line": 0, "replacement": "new\n"}],
        },
        _tool_context(tmp_path),
    )

    assert (root / "created.txt").read_text(encoding="utf-8") == "new\n"
    assert isinstance(result.output, dict)
    assert result.output["before_sha256"] is None
    assert result.output["line_count"] == 1


def test_file_patch_rejects_nul_replacement_without_changing_existing_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = "current\n"
    path = root / "sample.txt"
    path.write_text(original, encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run(
            {
                "path": "sample.txt",
                "expected_sha256": _sha256_text(original),
                "edits": [
                    {
                        "start_line": 1,
                        "end_line": 1,
                        "replacement": "text\x00binary\n",
                    }
                ],
            },
            _tool_context(tmp_path),
        )

    assert path.read_text(encoding="utf-8") == original


def test_file_patch_rejects_nul_replacement_when_creating_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "created.txt"
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run(
            {
                "path": "created.txt",
                "create_if_missing": True,
                "edits": [
                    {
                        "start_line": 1,
                        "end_line": 0,
                        "replacement": "text\x00binary\n",
                    }
                ],
            },
            _tool_context(tmp_path),
        )

    assert not path.exists()


@pytest.mark.parametrize("create_if_missing", ["false", "0", 1])
def test_file_patch_rejects_non_bool_create_if_missing(
    tmp_path: Path,
    create_if_missing: object,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="create_if_missing must be a boolean"):
        tool.run(
            {
                "path": "created.txt",
                "create_if_missing": create_if_missing,
                "edits": [{"start_line": 1, "end_line": 0, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )


def test_file_patch_rejects_expected_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.txt").write_text("current\n", encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="expected_sha256 does not match"):
        tool.run(
            {
                "path": "sample.txt",
                "expected_sha256": _sha256_text("old\n"),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )

    assert (root / "sample.txt").read_text(encoding="utf-8") == "current\n"


def test_file_patch_rejects_malformed_expected_hash(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.txt").write_text("current\n", encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="expected_sha256 must be a SHA-256 hex digest"):
        tool.run(
            {
                "path": "sample.txt",
                "expected_sha256": "not-a-sha",
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )


def test_file_patch_rejects_path_outside_write_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    writable = root / "writable"
    writable.mkdir()
    outside = root / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    tool = FilePatchTool(
        _file_config(root, patch_enabled=True, write_roots=(writable.resolve(),))
    )

    with pytest.raises(FileToolError, match="outside tools.files.write_roots"):
        tool.run(
            {
                "path": str(outside),
                "expected_sha256": _sha256_text("outside\n"),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )


def test_file_patch_rejects_symlink_ancestor_even_inside_write_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (real / "existing.txt").write_text("current\n", encoding="utf-8")
    subdir = real / "sub"
    subdir.mkdir()
    link = root / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="symlink ancestors are not patched"):
        tool.run(
            {
                "path": "link/existing.txt",
                "expected_sha256": _sha256_text("current\n"),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )
    with pytest.raises(FileToolError, match="symlink ancestors are not patched"):
        tool.run(
            {
                "path": "link/sub/new.txt",
                "create_if_missing": True,
                "edits": [{"start_line": 1, "end_line": 0, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )


def test_file_patch_rejects_absolute_path_with_symlink_ancestor_to_write_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    existing = real / "existing.txt"
    existing.write_text("current\n", encoding="utf-8")
    link = tmp_path / "link-to-real"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    tool = FilePatchTool(_file_config(real, patch_enabled=True, write_roots=(real.resolve(),)))

    with pytest.raises(FileToolError, match="symlink ancestors are not patched"):
        tool.run(
            {
                "path": str(link / "existing.txt"),
                "expected_sha256": _sha256_text("current\n"),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )

    assert existing.read_text(encoding="utf-8") == "current\n"


def test_file_patch_rejects_binary_and_symlink_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(b"text\x00binary")
    target = root / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match="binary files are not allowed"):
        tool.run(
            {
                "path": "binary.dat",
                "expected_sha256": hashlib.sha256(b"text\x00binary").hexdigest(),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )
    with pytest.raises(FileToolError, match="symlink files are not patched"):
        tool.run(
            {
                "path": "link.txt",
                "expected_sha256": _sha256_text("target\n"),
                "edits": [{"start_line": 1, "end_line": 1, "replacement": "new\n"}],
            },
            _tool_context(tmp_path),
        )


@pytest.mark.parametrize(
    ("edits", "match"),
    [
        (
            [
                {"start_line": 2, "end_line": 3, "replacement": "x\n"},
                {"start_line": 3, "end_line": 3, "replacement": "y\n"},
            ],
            "overlap",
        ),
        (
            [
                {"start_line": 3, "end_line": 3, "replacement": "x\n"},
                {"start_line": 2, "end_line": 2, "replacement": "y\n"},
            ],
            "original file order",
        ),
        (
            [{"start_line": 6, "end_line": 5, "replacement": "x\n"}],
            "insertion line must be at most",
        ),
        (
            [{"start_line": 0, "end_line": 0, "replacement": "x\n"}],
            "start_line must be at least 1",
        ),
    ],
)
def test_file_patch_rejects_overlap_and_invalid_edits(
    tmp_path: Path,
    edits: list[dict[str, object]],
    match: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = "one\ntwo\nthree\n"
    (root / "sample.txt").write_text(original, encoding="utf-8")
    tool = FilePatchTool(_file_config(root, patch_enabled=True, write_roots=(root.resolve(),)))

    with pytest.raises(FileToolError, match=match):
        tool.run(
            {
                "path": "sample.txt",
                "expected_sha256": _sha256_text(original),
                "edits": edits,
            },
            _tool_context(tmp_path),
        )


def test_default_registry_registers_file_tools_unless_disabled(tmp_path: Path) -> None:
    disabled = AlphaConfig(
        db_path=tmp_path / "disabled.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        file_tool=FileToolConfig(enabled=False, allowed_roots=(tmp_path,)),
    )
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
    patch_without_roots = AlphaConfig(
        db_path=tmp_path / "patch-without-roots.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        file_tool=FileToolConfig(
            enabled=True,
            allowed_roots=(tmp_path,),
            patch_enabled=True,
            write_roots=(),
        ),
    )

    assert build_tool_registry(disabled).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
    ]
    assert build_tool_registry(default_enabled).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_LIST_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
    ]
    assert build_tool_registry(patch_without_roots).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_LIST_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
    ]
    assert build_tool_registry(patch_enabled).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_LIST_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
        FILE_PATCH_TOOL_NAME,
    ]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
