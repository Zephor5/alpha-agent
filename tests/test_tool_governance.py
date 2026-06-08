from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from alpha_agent.config import BashToolConfig
from alpha_agent.llm.base import openai_compatible_tool_payload
from alpha_agent.runtime.tools import ToolExecutor
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import (
    ToolAvailability,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
    TurnToolState,
)
from alpha_agent.tools.bash import BashTool
from alpha_agent.tools.memory_propose import MemoryProposeTool
from alpha_agent.tools.memory_recall import MemoryRecallTool
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.tools.web_search import TavilyWebSearchTool


class _GovernedTool:
    spec = ToolSpec(
        name="governed",
        description="Return a small governed result.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        toolset="test",
        read_only=True,
        concurrency_safe=True,
        max_result_size_chars=128,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(name=self.spec.name, output="ok")


class _UnavailableTool:
    spec = ToolSpec(
        name="offline",
        description="Unavailable test tool.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        toolset="test",
        max_result_size_chars=64,
    )

    def __init__(self) -> None:
        self.ran = False

    def check_available(self) -> ToolAvailability:
        return ToolAvailability.unavailable(
            "missing test dependency",
            details={"dependency": "test"},
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        self.ran = True
        return ToolResult(name=self.spec.name, output="should not run")


class _LargeGovernedTool:
    spec = ToolSpec(
        name="large_governed",
        description="Return output larger than the declared result limit.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        toolset="test",
        max_result_size_chars=60,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(
            name=self.spec.name,
            output="0123456789" * 8,
            metadata={"source": "tool"},
        )


class _TinyLimitTool:
    spec = ToolSpec(
        name="tiny_limit",
        description="Return output larger than a too-small marker limit.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        toolset="test",
        max_result_size_chars=8,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(name=self.spec.name, output="abcdefghijklmnopqrstuvwxyz")


class _TinyUnavailableTool:
    spec = ToolSpec(
        name="tiny_offline",
        description="Unavailable test tool with a tiny model result limit.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        toolset="test",
        max_result_size_chars=128,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability.unavailable(
            "missing test dependency",
            details={
                "dependency": "test",
                "diagnostic": "x" * 500,
            },
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(name=self.spec.name, output="should not run")


class _BadSpecTool:
    spec = {
        "name": "bad_spec",
        "description": "Declare malformed tool spec.",
        "parameters": {},
    }

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(name="bad_spec", output="bad")


class _CountingTool:
    spec = ToolSpec(
        name="counting",
        description="Return the number of real dispatches.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
        toolset="test",
        max_result_size_chars=1000,
    )

    def __init__(self) -> None:
        self.run_count = 0

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del arguments
        self.run_count += 1
        return ToolResult(
            name=self.spec.name,
            output={"run_count": self.run_count},
            metadata={"turn_state_id": id(context.turn_state)},
        )


def test_registry_exposes_governance_metadata_and_filters_unavailable_tools() -> None:
    registry = ToolRegistry()
    registry.register(_GovernedTool())
    registry.register(_UnavailableTool())

    governed = registry.describe("governed")
    unavailable = registry.describe("offline")

    assert governed is not None
    assert governed.spec.to_dict() == {
        "name": "governed",
        "description": "Return a small governed result.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
        "toolset": "test",
        "read_only": True,
        "concurrency_safe": True,
        "destructive": False,
        "requires_user_interaction": False,
        "max_result_size_chars": 128,
    }
    assert governed.availability.available is True
    assert unavailable is not None
    assert unavailable.availability.to_dict() == {
        "available": False,
        "reason": "missing test dependency",
        "details": {"dependency": "test"},
    }
    assert registry.available_names() == ["governed"]

    definitions = registry.to_llm_tool_definitions()

    assert [definition.name for definition in definitions] == ["governed"]
    assert set(definitions[0].__dict__) == {"name", "description", "parameters", "strict"}
    function_payload = openai_compatible_tool_payload(definitions[0])["function"]
    assert set(function_payload) == {"name", "description", "parameters", "strict"}
    assert "toolset" not in function_payload
    assert "read_only" not in function_payload
    assert "max_result_size_chars" not in function_payload
    assert "group" not in governed.spec.to_dict()


def test_tool_execution_context_direct_construction_gets_turn_state(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path,
        check_canceled=lambda _stage: None,
    )

    assert isinstance(context.turn_state, TurnToolState)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "100"])
def test_tool_spec_rejects_non_positive_or_non_integer_limits(limit: Any) -> None:
    with pytest.raises(ValueError, match="max_result_size_chars"):
        ToolSpec(
            name="bad_limit",
            description="Bad limit.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            max_result_size_chars=limit,
        )


def test_registry_validates_tool_spec_on_register() -> None:
    registry = ToolRegistry()

    try:
        registry.register(cast(Any, _BadSpecTool()))
    except TypeError as exc:
        assert "ToolSpec" in str(exc)
    else:
        raise AssertionError("bad spec should be rejected")

    assert registry.names() == []


def test_existing_tools_declare_governance_metadata(tmp_path: Path) -> None:
    bash_tool = BashTool(
        config=BashToolConfig(
            enabled=True,
            default_workdir=tmp_path,
            allowed_workdirs=(tmp_path,),
            max_output_chars=4096,
        )
    )

    assert MemoryRecallTool().spec.to_dict() == {
        "name": "memory_recall",
        "description": MemoryRecallTool().spec.description,
        "parameters": MemoryRecallTool().spec.parameters,
        "strict": True,
        "toolset": "memory",
        "read_only": True,
        "concurrency_safe": True,
        "destructive": False,
        "requires_user_interaction": False,
        "max_result_size_chars": 100_000,
    }
    assert MemoryProposeTool().spec.toolset == "memory"
    assert MemoryProposeTool().spec.read_only is False
    assert MemoryProposeTool().spec.destructive is True
    assert bash_tool.spec.toolset == "shell"
    assert bash_tool.spec.destructive is True
    assert bash_tool.spec.requires_user_interaction is False
    assert bash_tool.spec.max_result_size_chars == 4096

    unavailable_web = TavilyWebSearchTool(api_key="")
    assert unavailable_web.spec.toolset == "web"
    assert unavailable_web.spec.read_only is True
    assert unavailable_web.check_available().to_dict() == {
        "available": False,
        "reason": "web search credentials are not configured",
    }


def test_executor_records_output_limit_contract_without_mutating_tool_result_metadata(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    registry.register(_GovernedTool())
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[ToolCall(id="call_1", name="governed", arguments={})],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )
    traces = store.list_runtime_traces("s1")

    assert executed[0].result.output == "ok"
    assert executed[0].result.metadata == {}
    assert traces[0].event_type == "tool.started"
    assert traces[0].metadata["tool_spec"]["max_result_size_chars"] == 128
    assert "tool_metadata" not in traces[0].metadata
    assert traces[1].event_type == "tool.completed"
    assert traces[1].content == "ok"
    assert traces[1].metadata["result"] == {
        "metadata": {},
        "name": "governed",
        "output": "ok",
    }
    assert traces[1].metadata["tool_output_limit"] == {
        "limit_chars": 128,
        "omitted_chars": 0,
        "original_chars": 2,
        "truncated": False,
    }


def test_executor_repeated_call_guard_is_turn_scoped_and_blocks_fourth_dispatch(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    tool = _CountingTool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry)
    turn_state = TurnToolState()

    def execute(arguments: dict[str, Any], call_id: str) -> ToolResult:
        return executor.execute(
            calls=[ToolCall(id=call_id, name="counting", arguments=arguments)],
            session_id="s1",
            turn_state=turn_state,
            write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
                session_id="s1",
                event_type=event_type,
                content=content,
                metadata=metadata,
            ),
            check_canceled=lambda _stage: None,
            recover_errors=True,
        )[0].result

    first = execute({"b": 2, "a": 1}, "call_1")
    second = execute({"a": 1, "b": 2}, "call_2")
    third = execute({"b": 2, "a": 1}, "call_3")
    fourth = execute({"a": 1, "b": 2}, "call_4")

    assert first.output == {"run_count": 1}
    assert second.output == {"run_count": 2}
    assert tool.run_count == 3
    assert isinstance(third.output, dict)
    third_output = cast(dict[str, Any], third.output)
    third_warning = cast(dict[str, Any], third_output["warning"])
    assert third_output["result"] == {"run_count": 3}
    assert third_warning["code"] == "repeated_tool_call"
    assert third_warning["repeat_count"] == 3
    assert third.metadata["repeated_tool_call"]["arguments_hash"]
    assert isinstance(fourth.output, dict)
    fourth_output = cast(dict[str, Any], fourth.output)
    fourth_error = cast(dict[str, Any], fourth_output["error"])
    fourth_details = cast(dict[str, Any], fourth_error["details"])
    assert fourth_error["code"] == "repeated_tool_call_blocked"
    assert fourth_details["repeat_count"] == 4
    assert fourth_details["arguments_hash"]
    assert "arguments" not in fourth_details
    assert tool.run_count == 3

    fresh_turn_result = executor.execute(
        calls=[ToolCall(id="call_5", name="counting", arguments={"a": 1, "b": 2})],
        session_id="s1",
        turn_state=TurnToolState(),
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )[0].result

    assert fresh_turn_result.output == {"run_count": 4}


def test_executor_truncates_oversized_output_before_trace_and_tool_message_boundary(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    registry.register(_LargeGovernedTool())
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[ToolCall(id="call_1", name="large_governed", arguments={})],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )
    completed_trace = store.list_runtime_traces("s1")[1]
    full_output = "0123456789" * 8
    marker = "[tool output truncated: 62 chars omitted]"
    preview = f"{full_output[:18]}\n{marker}"

    assert executed[0].result.output == full_output
    assert executed[0].result.metadata == {"source": "tool"}
    assert completed_trace.content == preview
    assert marker in completed_trace.content
    assert len(completed_trace.content) <= 60
    assert completed_trace.metadata["result"] == {
        "metadata": {"source": "tool"},
        "name": "large_governed",
        "output": preview,
    }
    assert completed_trace.metadata["tool_output_limit"] == {
        "limit_chars": 60,
        "omitted_chars": 62,
        "original_chars": 80,
        "truncated": True,
    }
    assert full_output not in str(completed_trace.metadata)


def test_executor_uses_raw_prefix_when_limit_cannot_fit_truncation_marker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    registry.register(_TinyLimitTool())
    executor = ToolExecutor(registry)

    executor.execute(
        calls=[ToolCall(id="call_1", name="tiny_limit", arguments={})],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )
    completed_trace = store.list_runtime_traces("s1")[1]

    assert completed_trace.content == "abcdefgh"
    assert "[tool output truncated:" not in completed_trace.content
    assert len(completed_trace.content) <= 8
    assert completed_trace.metadata["result"]["output"] == "abcdefgh"
    assert completed_trace.metadata["tool_output_limit"] == {
        "limit_chars": 8,
        "omitted_chars": 18,
        "original_chars": 26,
        "truncated": True,
    }
    assert "abcdefghijklmnopqrstuvwxyz" not in str(completed_trace.metadata)


def test_executor_returns_structured_unavailable_tool_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    tool = _UnavailableTool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[ToolCall(id="call_1", name="offline", arguments={})],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )
    traces = store.list_runtime_traces("s1")
    output = executed[0].result.output

    assert tool.ran is False
    assert isinstance(output, dict)
    error = cast(dict[str, Any], output["error"])
    assert error["code"] == "tool_unavailable"
    assert error["details"]["availability"] == {
        "available": False,
        "reason": "missing test dependency",
        "details": {"dependency": "test"},
    }
    assert executed[0].result.metadata == {
        "failed": True,
        "error": "Tool unavailable: offline: missing test dependency",
        "error_type": "ToolUnavailableError",
        "tool_call_id": "call_1",
    }
    assert [trace.event_type for trace in traces] == ["tool.started", "tool.failed"]
    assert traces[0].metadata["tool_availability"]["available"] is False
    failed_payload = json.loads(traces[1].content)
    assert failed_payload["error"]["code"] == "tool_unavailable"
    assert len(traces[1].content) <= 64
    assert traces[1].metadata["result"]["metadata"] == executed[0].result.metadata
    assert traces[1].metadata["result"]["output"] == failed_payload
    assert traces[1].metadata["tool_output_limit"]["limit_chars"] == 64


def test_executor_keeps_small_limit_structured_unavailable_content_valid_json(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    registry.register(_TinyUnavailableTool())
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[ToolCall(id="call_1", name="tiny_offline", arguments={})],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )
    failed_trace = store.list_runtime_traces("s1")[1]

    assert executed[0].result.metadata["error_type"] == "ToolUnavailableError"
    payload = json.loads(failed_trace.content)
    assert payload["error"]["code"] == "tool_unavailable"
    assert payload["error"]["message"] == "Tool unavailable: tiny_offline: missing test dependency"
    assert failed_trace.metadata["result"]["output"] == payload
    assert failed_trace.metadata["tool_output_limit"]["limit_chars"] == 128
    assert failed_trace.metadata["tool_output_limit"]["truncated"] is True
    assert failed_trace.metadata["tool_output_limit"]["structured"] is True
    assert "diagnostic" not in failed_trace.content
    assert "x" * 100 not in failed_trace.content


def _store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store
