from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from typer.testing import CliRunner

from alpha_agent import cli as cli_module
from alpha_agent.cli import app
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.runtime.prompt_builder import build_answer_prompt_messages
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.state.store import StateStore
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME
from alpha_agent.utils.system_reminder import SYSTEM_REMINDER_OPEN
from tests.cognition.test_belief_projection_apply import belief


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


def test_init_creates_state_database_without_loading_long_term_records(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["init"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Initialized Alpha Agent database" in result.output
    with StateStore(tmp_path / "alpha.db").connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert tables == {
            "sessions",
            "session_messages",
            "session_counterparts",
            "session_summary_snapshots",
            "runtime_traces",
            "gateway_session_mappings",
            "gateway_dedup",
            "cognitive_events",
            "counterpart_view",
            "atomic_beliefs",
            "summary_beliefs",
            "belief_entity_index",
            "belief_about_index",
            "belief_search_terms_fts",
            "belief_search_terms_fts_config",
            "belief_search_terms_fts_content",
            "belief_search_terms_fts_data",
            "belief_search_terms_fts_docsize",
            "belief_search_terms_fts_idx",
            "belief_search_trigram_fts",
            "belief_search_trigram_fts_config",
            "belief_search_trigram_fts_content",
            "belief_search_trigram_fts_data",
            "belief_search_trigram_fts_docsize",
            "belief_search_trigram_fts_idx",
            "cognition_state_audit",
            "background_source_progress",
            "background_source_window",
            "background_stage_run",
            "goal_view",
            "subject_view",
            "cognition_worker_checkpoint",
        }


def test_debug_prompt_renders_minimal_prompt_for_existing_session(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="hi",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Message 1 [system]" in result.output
    assert "Identity: Alpha Agent" in result.output
    assert "hello" in result.output
    assert "hi" in result.output
    assert "continue" in result.output


def test_debug_prompt_filters_session_profile_reminder_by_default(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_reminder(
        session_id="s1",
        raw_content="Counterpart profile: Stable debug profile.",
        reminder_type="counterpart_profile",
    )
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Message 1 [system]" in result.output
    assert "Counterpart profile: Stable debug profile." not in result.output
    assert "hello" in result.output


def test_debug_prompt_can_include_raw_reminders_explicitly(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_reminder(
        session_id="s1",
        raw_content="Counterpart profile: Stable debug profile.",
        reminder_type="counterpart_profile",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1", "--include-reminders"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert SYSTEM_REMINDER_OPEN in result.output
    assert "Counterpart profile: Stable debug profile." in result.output


def test_normal_chat_history_display_filters_raw_reminders(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    reminder = store.append_session_reminder(
        session_id="s1",
        raw_content="Counterpart profile: Stable display profile.",
        reminder_type="counterpart_profile",
    )
    user_message = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    assistant_message = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="hi",
    )

    display_messages = cli_module._displayable_session_messages(store, "s1")

    assert display_messages == [user_message, assistant_message]
    assert reminder in store.list_session_messages("s1")
    assert reminder.kind == "system_reminder"
    assert SYSTEM_REMINDER_OPEN in reminder.raw_content
    assert "Counterpart profile: Stable display profile." in reminder.raw_content


def test_debug_prompt_matches_shared_answer_prompt_builder(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="hi",
    )
    expected_messages = build_answer_prompt_messages(
        session_history=SessionContextAssembler(store).load("s1").chat_messages,
        current_turn_messages=[{"role": "user", "content": "continue"}],
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    header_positions = []
    for index, expected in enumerate(expected_messages, start=1):
        header_positions.append(result.output.index(f"Message {index} [{expected['role']}]"))
        content = str(expected.get("content") or "")
        if expected["role"] != "system":
            assert content in result.output
    assert header_positions == sorted(header_positions)
    assert "Identity: Alpha Agent" in result.output
    assert f"Message {len(expected_messages) + 1} [" not in result.output


def test_debug_prompt_uses_latest_compressed_boundary(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    user = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="old source before compressed boundary",
    )
    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="old answer before compressed boundary",
    )
    assert assistant.ordinal == user.ordinal + 1
    store.append_compressed_message(
        session_id="s1",
        raw_content="latest compressed handover",
        compression_point_ordinal=assistant.ordinal,
        compression_version="test-v1",
    )
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="fresh source after compressed boundary",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Identity: Alpha Agent" in result.output
    assert "latest compressed handover" in result.output
    assert "fresh source after compressed boundary" in result.output
    assert "continue" in result.output
    assert "old source before compressed boundary" not in result.output
    assert "old answer before compressed boundary" not in result.output


def test_debug_prompt_trace_renders_recent_cognitive_events(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    AlphaAgent(store=store, llm_provider=MockLLMProvider()).respond("hello", session_id="s1")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1", "--trace"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Cognitive Trace" in result.output
    for kind in [
        "perceived",
        "acted",
        "turn_sources_recorded",
    ]:
        assert f"kind={kind}" in result.output
    for retired_kind in [
        "attended",
        "interpreted",
        "judged",
        "decided",
        "received_feedback",
        "reflected",
        "revised",
    ]:
        assert f"kind={retired_kind}" not in result.output
    assert "turn_id=" in result.output
    assert "session_id=s1" in result.output
    assert "tick" + "_id=" not in result.output


def test_debug_prompt_trace_summarizes_memory_tool_results(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    BeliefProjection(store).upsert_atomic(
        belief(
            "belief:python",
            "User prefers Python examples.",
            object_="Python examples",
        )
    )
    AlphaAgent(store=store, llm_provider=_MemoryTraceProvider()).respond(
        "Actually use Rust examples.",
        session_id="s1",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1", "--trace"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Memory Tool Trace" in result.output
    assert "tool=memory_recall" in result.output
    assert "results=belief:python" in result.output
    assert "tool=memory_propose" in result.output
    assert "status=needs_target_selection" in result.output
    assert "next_action=review_candidates" in result.output
    assert "updates=append_distinct:needs_target_selection" in result.output
    assert "candidates=belief:python" in result.output


def test_skills_list_reads_builtin_skills_without_state_store(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["skills", "list"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Skill:" in result.output


class _MemoryTraceProvider:
    name = "memory-trace-provider"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: object | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, response_format
        self.calls += 1
        if self.calls == 1:
            arguments = {
                "query": "Python examples",
                "scope": "global",
                "max_results": 4,
            }
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_recall",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, sort_keys=True),
                    )
                ],
            )
        if self.calls == 2:
            arguments = {
                "updates": [
                    {
                        "operation": "append_distinct",
                        "target_belief_ids": [],
                        "reviewed_candidate_ids": [],
                        "target_hint": "Python examples preference",
                        "memory": {
                            "type": "preference",
                            "content": "User prefers Rust examples.",
                            "evidence": "User said: actually use Rust examples now.",
                            "scope": "global",
                        },
                        "reason": (
                            "The user expressed a related but non-identical "
                            "example-language preference."
                        ),
                    }
                ]
            }
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_memory",
                        name=MEMORY_PROPOSE_TOOL_NAME,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, sort_keys=True),
                    )
                ],
            )
        return LLMResponse(
            content="Recorded the memory tool result.",
            model="test",
            provider=self.name,
        )
