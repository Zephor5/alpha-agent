from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.daemon.conversation_import import ConversationImportService
from alpha_agent.state.store import StateStore


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


def _deepseek_export() -> list[dict[str, object]]:
    return [
        {
            "id": "deepseek_conv",
            "title": "CLI conversion",
            "inserted_at": "2026-01-01T10:00:00.000000+08:00",
            "updated_at": "2026-01-01T10:03:00.000000+08:00",
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": None,
                    "children": ["1"],
                    "message": None,
                },
                "1": {
                    "id": "1",
                    "parent": "root",
                    "children": ["2"],
                    "message": {
                        "files": [],
                        "model": "deepseek-chat",
                        "inserted_at": "2026-01-01T10:01:00.000000+08:00",
                        "fragments": [{"type": "REQUEST", "content": "hello"}],
                    },
                },
                "2": {
                    "id": "2",
                    "parent": "1",
                    "children": [],
                    "message": {
                        "files": [],
                        "model": "deepseek-chat",
                        "inserted_at": "2026-01-01T10:01:00.000000+08:00",
                        "fragments": [{"type": "RESPONSE", "content": "hi"}],
                    },
                },
            },
        }
    ]


def _deepseek_export_with_two_conversations() -> list[dict[str, object]]:
    source = _deepseek_export()
    second = json.loads(json.dumps(source[0]))
    second["id"] = "deepseek_conv_2"
    second["title"] = "CLI conversion 2"
    source.append(second)
    return source


def _write_deepseek_export(path: Path, payload: object | None = None) -> None:
    content = json.dumps(payload if payload is not None else _deepseek_export())
    path.write_text(content, encoding="utf-8")


def _assert_import_validator_accepts(path: Path, tmp_path: Path) -> None:
    store = StateStore(tmp_path / "validator.db")
    store.initialize()
    summary = ConversationImportService(store).import_payload(
        path.read_text(encoding="utf-8"),
        input_name=path.name,
        dry_run=True,
    )
    assert summary.source_provider == "deepseek"
    assert summary.messages_inserted == 2


def test_cognition_import_convert_deepseek_writes_normalized_file(tmp_path: Path) -> None:
    source_path = tmp_path / "deepseek.json"
    output_path = tmp_path / "normalized.json"
    _write_deepseek_export(source_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "convert", "deepseek", str(source_path), str(output_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "conversion source_provider=deepseek conversations=1 messages=2" in result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_provider"] == "deepseek"
    assert payload["timezone"] == "+08:00"
    assert payload["conversations"][0]["messages"][1]["created_at"].endswith("000001+08:00")
    _assert_import_validator_accepts(output_path, tmp_path)


def test_cognition_import_convert_deepseek_limits_conversation_count(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "deepseek.json"
    output_path = tmp_path / "normalized.json"
    _write_deepseek_export(source_path, _deepseek_export_with_two_conversations())
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cognition",
            "import",
            "convert",
            "deepseek",
            str(source_path),
            str(output_path),
            "--limit",
            "1",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "conversion source_provider=deepseek conversations=1 messages=2" in result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["external_conversation_id"] for item in payload["conversations"]] == [
        "deepseek_conv"
    ]
    _assert_import_validator_accepts(output_path, tmp_path)


def test_cognition_import_convert_deepseek_rejects_existing_output_without_force(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "deepseek.json"
    output_path = tmp_path / "normalized.json"
    _write_deepseek_export(source_path)
    output_path.write_text("keep me", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "convert", "deepseek", str(source_path), str(output_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Output file already exists. Use --force to overwrite." in result.output
    assert output_path.read_text(encoding="utf-8") == "keep me"


def test_cognition_import_convert_deepseek_force_overwrites_existing_output(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "deepseek.json"
    output_path = tmp_path / "normalized.json"
    _write_deepseek_export(source_path)
    output_path.write_text("replace me", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "cognition",
            "import",
            "convert",
            "deepseek",
            str(source_path),
            str(output_path),
            "--force",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["source_provider"] == "deepseek"


def test_cognition_import_convert_deepseek_renders_conversion_errors(tmp_path: Path) -> None:
    source_path = tmp_path / "deepseek.json"
    output_path = tmp_path / "normalized.json"
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    node = mapping["1"]
    assert isinstance(node, dict)
    message = node["message"]
    assert isinstance(message, dict)
    fragments = message["fragments"]
    assert isinstance(fragments, list)
    fragments.append({"type": "AUDIO", "content": "unsupported"})
    _write_deepseek_export(source_path, source)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "convert", "deepseek", str(source_path), str(output_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Invalid DeepSeek conversation export." in result.output
    assert "unsupported fragment type" in result.output
    assert not output_path.exists()
