from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.memory.controller import MemoryController
from alpha_agent.memory.extractor import (
    ExtractionSchemaError,
    LLMAssistedMemoryExtractor,
    MemoryExtractionContext,
)
from alpha_agent.memory.models import MemoryScope
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from tests.memory_eval import ExpectedCandidate, assert_extraction_candidates


def test_extraction_eval_covers_preference_fact_and_procedure() -> None:
    assert_extraction_candidates(
        "I prefer concise answers",
        [
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="User prefers: concise answers",
                subject="user",
                predicate="prefers",
                object_value="concise answers",
                min_confidence=0.7,
            )
        ],
    )
    assert_extraction_candidates(
        "my favorite editor is neovim",
        [
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="user.favorite_editor is neovim",
                subject="user.favorite_editor",
                predicate="is",
                object_value="neovim",
            )
        ],
    )
    assert_extraction_candidates(
        "when I ask you to debug tests, reproduce the failure first",
        [
            ExpectedCandidate(
                candidate_type="procedural_candidate",
                content_contains="reproduce the failure first",
                subject="user",
                predicate="procedure",
                object_value="debug tests",
            )
        ],
    )


def test_extraction_eval_documents_do_not_remember_gap() -> None:
    candidates = assert_extraction_candidates(
        "do not remember that I prefer tea",
        [],
    )
    assert candidates == []


def test_llm_assisted_extractor_validates_strict_candidate_schema() -> None:
    provider = _StaticJSONProvider(
        """
        {
          "candidates": [
            {
              "layer": "semantic",
              "memory_type": "preference",
              "content": "User prefers: tea",
              "entities": ["user", "tea"],
              "weak_structure": {
                "subject": "user",
                "predicate": "prefers",
                "object": "tea"
              },
              "confidence": 0.82,
              "stability": 0.74,
              "salience": 0.9,
              "source_ids": ["msg-user"],
              "sensitivity_flags": [],
              "rationale": "Explicit preference statement."
            }
          ]
        }
        """
    )

    candidates = LLMAssistedMemoryExtractor(provider).extract(
        user_message="remember that I prefer tea",
        assistant_response="ok",
        source_event_ids=["msg-user", "msg-assistant"],
    )

    assert len(candidates) == 1
    assert candidates[0].type == "semantic"
    assert candidates[0].metadata["memory_type"] == "preference"
    assert candidates[0].metadata["stability"] == 0.74
    assert candidates[0].metadata["rationale"] == "Explicit preference statement."


def test_llm_assisted_extractor_rejects_invalid_json_schema() -> None:
    provider = _StaticJSONProvider(
        """
        {
          "candidates": [
            {
              "layer": "semantic",
              "content": "missing required fields",
              "confidence": 2.0
            }
          ]
        }
        """
    )

    with pytest.raises(ExtractionSchemaError):
        LLMAssistedMemoryExtractor(provider).extract(
            user_message="remember that I prefer tea",
            assistant_response="ok",
            source_event_ids=["msg-user"],
        )


def test_controller_extraction_policy_blocks_sensitive_and_group_ambient_writes(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    controller = MemoryController(store, retriever=MemoryRetriever(store))
    group_scope = MemoryScope.from_source_metadata(
        session_id="s1",
        source_metadata={
            "platform": "slack",
            "chat_id": "team",
            "chat_type": "group",
            "user_id": "u1",
        },
    )

    sensitive = controller.extract_candidates(
        session_id="s1",
        user_message="remember that my password is swordfish",
        assistant_response="ok",
        source_message_ids=[],
        scope=MemoryScope.default(),
    )
    ambient_group = controller.extract_candidates(
        session_id="s1",
        user_message="I prefer tea",
        assistant_response="ok",
        source_message_ids=[],
        scope=group_scope,
    )
    explicit_group = controller.extract_candidates(
        session_id="s1",
        user_message="remember that I prefer tea",
        assistant_response="ok",
        source_message_ids=[],
        scope=group_scope,
    )

    assert sensitive == []
    assert ambient_group == []
    assert [candidate.candidate_type for candidate in explicit_group] == [
        "semantic",
        "episodic",
    ]


def test_controller_extraction_policy_blocks_platform_system_sources(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    system_message = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="remember that I prefer tea",
        source_metadata={"is_system_message": True},
    )
    controller = MemoryController(store, retriever=MemoryRetriever(store))

    candidates = controller.extract_candidates(
        session_id="s1",
        user_message="remember that I prefer tea",
        assistant_response="ok",
        source_message_ids=[system_message.id],
        scope=MemoryScope.default(),
    )

    assert candidates == []


def test_controller_passes_session_and_retrieved_memory_context_to_extractor(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    existing = SemanticMemoryManager(store).upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers: tea",
        scope=MemoryScope.default(),
    )
    context = MemoryRetriever(store).retrieve_context(
        "actually I prefer coffee",
        "s1",
        scopes=MemoryScope.default().allowed_read_scopes(),
        record_access=False,
    )
    extractor = _RecordingExtractor()
    controller = MemoryController(
        store,
        retriever=MemoryRetriever(store),
        extractor=extractor,
    )

    controller.extract_candidates(
        session_id="s1",
        user_message="actually I prefer coffee",
        assistant_response="ok",
        source_message_ids=["msg-user"],
        scope=MemoryScope.default(),
        retrieved_context=context,
    )

    assert extractor.context is not None
    assert [memory.id for memory in extractor.context.active_semantic_memories] == [
        existing.id
    ]


class _StaticJSONProvider:
    name = "static-json"

    def __init__(self, content: str):
        self.content = content
        self.messages: list[ChatMessage] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> LLMResponse:
        del tools, tool_choice
        self.messages = messages
        return LLMResponse(content=self.content, model="test", provider=self.name)


class _RecordingExtractor:
    context: MemoryExtractionContext | None = None

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
        *,
        context: MemoryExtractionContext | None = None,
    ) -> list[Any]:
        del user_message, assistant_response, source_event_ids
        self.context = context
        return []
