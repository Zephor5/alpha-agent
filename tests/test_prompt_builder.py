from __future__ import annotations

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
    prompt = messages[1]["content"]

    assert "## Working Memory" in prompt
    assert "## Relevant User Facts" in prompt
    assert "## Relevant Episodes" in prompt
    assert "## Relevant Skills" in prompt
    assert "## Current User Message" in prompt
