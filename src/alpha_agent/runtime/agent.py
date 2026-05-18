"""Explicit personal agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.llm.base import LLMProvider
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import RetrievedContext
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.runtime.events import create_event
from alpha_agent.runtime.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one agent turn."""

    response: str
    session_id: str
    debug: dict[str, Any] = field(default_factory=dict)


class AlphaAgent:
    """Controllable synchronous agent runtime with explicit memory steps."""

    def __init__(
        self,
        store: MemoryStore,
        llm_provider: LLMProvider,
        working_memory: WorkingMemoryManager,
        retriever: MemoryRetriever,
        retrieval_limit: int = 8,
        prompt_builder: PromptBuilder | None = None,
        extractor: MemoryExtractor | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.working_memory = working_memory
        self.retriever = retriever
        self.retrieval_limit = retrieval_limit
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.extractor = extractor or MemoryExtractor()
        self.episodic = EpisodicMemoryManager(store)
        self.semantic = SemanticMemoryManager(store)
        self.procedural = ProceduralMemoryManager(store)

    def respond(self, user_message: str, session_id: str) -> AgentTurnResult:
        """Run one explicit agent turn."""

        user_event = self.store.insert_event(create_event(session_id, "user", user_message))
        self.working_memory.add_active_context(
            session_id=session_id,
            content=f"User: {user_message}",
            source_event_id=user_event.id,
            priority=0.6,
        )
        context = self.retriever.retrieve_context(
            user_message,
            session_id,
            limit=self.retrieval_limit,
        )
        messages = self.prompt_builder.build(user_message, context)
        llm_response = self.llm_provider.complete(messages)
        assistant_event = self.store.insert_event(
            create_event(session_id, "assistant", llm_response.content)
        )
        self.working_memory.add_active_context(
            session_id=session_id,
            content=f"Assistant: {llm_response.content}",
            source_event_id=assistant_event.id,
            priority=0.45,
        )
        candidates = self.extractor.extract(
            user_message=user_message,
            assistant_response=llm_response.content,
            source_event_ids=[user_event.id, assistant_event.id],
        )
        for candidate in candidates:
            if candidate.type == "episodic":
                self.episodic.create(
                    content=candidate.content,
                    source_event_ids=candidate.source_event_ids,
                    salience=candidate.salience,
                    confidence=candidate.confidence,
                )
            elif candidate.type == "semantic" and candidate.subject and candidate.predicate:
                self.semantic.upsert_fact(
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    object_value=candidate.object or "",
                    content=candidate.content,
                    confidence=candidate.confidence,
                    salience=candidate.salience,
                    source_memory_ids=candidate.source_event_ids,
                )
            elif candidate.type == "procedural_candidate":
                self.episodic.create(
                    content=f"Procedural candidate: {candidate.content}",
                    source_event_ids=candidate.source_event_ids,
                    salience=candidate.salience,
                    confidence=candidate.confidence,
                )

        return AgentTurnResult(
            response=llm_response.content,
            session_id=session_id,
            debug={
                "retrieved_memory_ids": self._retrieved_ids(context),
                "prompt_token_estimate": self.prompt_builder.rough_token_estimate(messages),
                "provider": llm_response.provider,
                "extracted_memory_count": len(candidates),
            },
        )

    def _retrieved_ids(self, context: RetrievedContext) -> dict[str, list[str]]:
        return {
            "working": [item.id for item in context.working_memory],
            "episodic": [item.id for item in context.episodic_memories],
            "semantic": [item.id for item in context.semantic_memories],
            "procedural": [item.id for item in context.procedural_memories],
        }
