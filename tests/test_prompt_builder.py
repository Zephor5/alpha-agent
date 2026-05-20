from __future__ import annotations

from typing import cast

from alpha_agent.llm.base import ChatMessage
from alpha_agent.memory.models import (
    EpisodicMemory,
    ProceduralMemory,
    RetrievedContext,
    SemanticMemory,
    WorkingMemoryItem,
)
from alpha_agent.runtime.prompt_builder import PromptBuilder


def test_prompt_includes_memory_sections() -> None:
    context = RetrievedContext(
        working_memory=[
            WorkingMemoryItem(
                id="wm1",
                session_id="s1",
                content="Current task: build MVP",
                source_event_id=None,
                priority=0.8,
                expires_at=None,
                created_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        semantic_memories=[
            SemanticMemory(
                id="sem1",
                subject="user",
                predicate="prefers",
                object="concise answers",
                content="User prefers concise answers",
                confidence=0.9,
                salience=0.8,
                source_memory_ids=[],
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        episodic_memories=[
            EpisodicMemory(
                id="epi1",
                content="User asked for memory MVP",
                summary="User asked for memory MVP",
                source_event_ids=[],
                people=[],
                places=[],
                topics=[],
                salience=0.8,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Debug failures",
                trigger="debug",
                procedure_markdown="1. Reproduce\n2. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )

    messages = PromptBuilder().build("What should we do?", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert [message["role"] for message in messages] == ["system", "system", "user"]
    assert messages[-1]["content"] == "What should we do?"
    assert "## Retrieved Context (Reference Only)" in context_prompt
    assert "### Recent Session Context" in context_prompt
    assert "### User Facts" in context_prompt
    assert "### Prior Episodes" in context_prompt
    assert "### Relevant Procedures" in context_prompt
    assert "## Current User Message" not in context_prompt


def test_prompt_omits_empty_context_message() -> None:
    context = RetrievedContext(
        working_memory=[],
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )

    messages = PromptBuilder().build("hello", context)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "hello"


def test_prompt_keeps_irrelevant_procedure_body_out_of_context() -> None:
    context = RetrievedContext(
        working_memory=[],
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Diagnose failures",
                trigger="debug failure regression",
                procedure_markdown="1. Reproduce\n2. Inspect\n3. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )

    messages = PromptBuilder().build("What tools do you have?", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert "Debug Loop: Diagnose failures" in context_prompt
    assert "1. Reproduce" not in context_prompt


def test_prompt_includes_matching_procedure_body() -> None:
    context = RetrievedContext(
        working_memory=[],
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Diagnose failures",
                trigger="debug failure regression",
                procedure_markdown="1. Reproduce\n2. Inspect\n3. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )

    messages = PromptBuilder().build("Debug this failing regression", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert "1. Reproduce" in context_prompt


def test_rough_token_estimate_tolerates_tool_call_assistant_messages() -> None:
    messages: list[ChatMessage] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_memory", "arguments": '{"query":"hello"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]

    assert PromptBuilder().rough_token_estimate(messages) == len("helloresult") // 4
