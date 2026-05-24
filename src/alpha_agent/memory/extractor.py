"""Post-turn memory extraction interfaces and implementations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from alpha_agent.llm.base import ChatMessage, LLMProvider
from alpha_agent.memory.models import ConversationMessage, ExtractedMemoryCandidate, SemanticMemory
from alpha_agent.memory.salience import SalienceScorer


class ExtractionSchemaError(ValueError):
    """Raised when an LLM extractor response fails the strict candidate schema."""


@dataclass(frozen=True)
class MemoryExtractionContext:
    """Context available to extractors without changing the runtime caller contract."""

    session_id: str
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    active_semantic_memories: list[SemanticMemory] = field(default_factory=list)


class MemoryExtractorInterface(Protocol):
    """Common interface for deterministic and LLM-assisted extractors."""

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
        *,
        context: MemoryExtractionContext | None = None,
    ) -> list[ExtractedMemoryCandidate]:
        """Extract candidate memories from a completed turn."""


class _WeakStructureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None


class _CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    layer: Literal["episodic", "semantic", "procedural"]
    memory_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    entities: list[str]
    weak_structure: _WeakStructureModel
    confidence: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    salience: float = Field(ge=0.0, le=1.0)
    source_ids: list[str]
    sensitivity_flags: list[str]
    rationale: str = Field(min_length=1)


class _ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidates: list[_CandidateModel]


class MemoryExtractor:
    """Extract transparent candidate memories without an LLM."""

    preference_patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bi prefer (?P<object>.+)", re.IGNORECASE), "prefers"),
        (re.compile(r"\bi like (?P<object>.+)", re.IGNORECASE), "likes"),
        (re.compile(r"\bi don't like (?P<object>.+)", re.IGNORECASE), "dislikes"),
    )
    durable_fact_patterns = (
        re.compile(r"\bmy (?P<subject>[a-zA-Z][\w -]{1,40}) is (?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bi am (?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bi work (?:at|for) (?P<object>.+)", re.IGNORECASE),
    )
    procedural_patterns = (
        re.compile(r"\bwhen i ask (?:you )?to (?P<trigger>.+), (?P<procedure>.+)", re.IGNORECASE),
        re.compile(r"\bfrom now on,? (?P<procedure>.+)", re.IGNORECASE),
    )

    def __init__(self, salience_scorer: SalienceScorer | None = None):
        self.salience_scorer = salience_scorer or SalienceScorer()

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
        *,
        context: MemoryExtractionContext | None = None,
    ) -> list[ExtractedMemoryCandidate]:
        """Extract candidate memories from a completed turn."""

        del context
        del assistant_response
        if _explicit_do_not_remember(user_message):
            return []

        candidates: list[ExtractedMemoryCandidate] = []
        salience = self.salience_scorer.score(user_message)

        for pattern, predicate in self.preference_patterns:
            match = pattern.search(user_message)
            if match:
                value = self._clean_value(match.group("object"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="semantic",
                        content=f"User {predicate}: {value}",
                        subject="user",
                        predicate=predicate,
                        object=value,
                        salience=max(salience, 0.85),
                        confidence=0.7,
                        source_event_ids=source_event_ids,
                        metadata={
                            "extractor": "preference_pattern",
                            "memory_type": "preference",
                            "entities": ["user", value],
                            "stability": 0.7,
                            "sensitivity_flags": [],
                            "rationale": "matched preference pattern",
                        },
                    )
                )
                break

        for pattern in self.durable_fact_patterns:
            match = pattern.search(user_message)
            if match:
                if "subject" in match.groupdict():
                    subject = f"user.{self._clean_value(match.group('subject')).replace(' ', '_')}"
                    predicate = "is"
                    value = self._clean_value(match.group("object"))
                elif "work" in pattern.pattern:
                    subject = "user"
                    predicate = "works_at"
                    value = self._clean_value(match.group("object"))
                else:
                    subject = "user"
                    predicate = "is"
                    value = self._clean_value(match.group("object"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="semantic",
                        content=f"{subject} {predicate} {value}",
                        subject=subject,
                        predicate=predicate,
                        object=value,
                        salience=max(salience, 0.65),
                        confidence=0.65,
                        source_event_ids=source_event_ids,
                        metadata={
                            "extractor": "durable_fact_pattern",
                            "memory_type": "fact",
                            "entities": ["user", value],
                            "stability": 0.65,
                            "sensitivity_flags": [],
                            "rationale": "matched durable fact pattern",
                        },
                    )
                )
                break

        lowered = user_message.lower()
        if "remember" in lowered or "important" in lowered or "actually" in lowered:
            candidates.append(
                ExtractedMemoryCandidate(
                    type="episodic",
                    content=f"User said: {user_message}",
                    salience=max(salience, 0.75),
                    confidence=0.7,
                    source_event_ids=source_event_ids,
                    metadata={
                        "extractor": "explicit_or_correction",
                        "memory_type": "episode",
                        "entities": ["user"],
                        "stability": 0.5,
                        "sensitivity_flags": [],
                        "rationale": "explicit memory or correction cue",
                    },
                )
            )

        for pattern in self.procedural_patterns:
            match = pattern.search(user_message)
            if match:
                procedure = self._clean_value(match.groupdict().get("procedure", user_message))
                trigger = self._clean_value(match.groupdict().get("trigger", "user instruction"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="procedural_candidate",
                        content=procedure,
                        subject="user",
                        predicate="procedure",
                        object=trigger,
                        salience=max(salience, 0.75),
                        confidence=0.55,
                        source_event_ids=source_event_ids,
                        metadata={
                            "extractor": "procedural_pattern",
                            "trigger": trigger,
                            "memory_type": "procedure",
                            "entities": ["user"],
                            "stability": 0.6,
                            "sensitivity_flags": [],
                            "rationale": "matched procedural instruction pattern",
                        },
                    )
                )
                break

        return candidates

    def _clean_value(self, value: str) -> str:
        return value.strip().rstrip(".!").strip()


class LLMAssistedMemoryExtractor:
    """Extract candidates with an injected LLM provider and strict JSON validation."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
        *,
        context: MemoryExtractionContext | None = None,
    ) -> list[ExtractedMemoryCandidate]:
        """Extract candidates from provider JSON; never performs network itself."""

        response = self.provider.complete(
            _extraction_messages(
                user_message=user_message,
                assistant_response=assistant_response,
                source_event_ids=source_event_ids,
                context=context,
            )
        )
        try:
            payload = json.loads(response.content)
            extracted = _ExtractionModel.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ExtractionSchemaError("LLM memory extraction response is invalid") from exc

        allowed_source_ids = set(source_event_ids)
        candidates: list[ExtractedMemoryCandidate] = []
        for item in extracted.candidates:
            unknown_sources = [
                source_id
                for source_id in item.source_ids
                if source_id not in allowed_source_ids
            ]
            if unknown_sources:
                raise ExtractionSchemaError(
                    "LLM memory extraction referenced unknown source ids: "
                    + ", ".join(unknown_sources)
                )
            candidate_type = (
                "procedural_candidate" if item.layer == "procedural" else item.layer
            )
            weak = item.weak_structure
            candidates.append(
                ExtractedMemoryCandidate(
                    type=candidate_type,
                    content=item.content.strip(),
                    salience=item.salience,
                    confidence=item.confidence,
                    stability=item.stability,
                    entities=list(item.entities),
                    subject=_optional_clean(weak.subject),
                    predicate=_optional_clean(weak.predicate),
                    object=_optional_clean(weak.object),
                    source_event_ids=list(item.source_ids),
                    metadata={
                        "extractor": "llm_assisted",
                        "memory_type": item.memory_type,
                        "entities": list(item.entities),
                        "stability": item.stability,
                        "sensitivity_flags": list(item.sensitivity_flags),
                        "rationale": item.rationale,
                    },
                )
            )
        return candidates


def _extraction_messages(
    *,
    user_message: str,
    assistant_response: str,
    source_event_ids: list[str],
    context: MemoryExtractionContext | None,
) -> list[ChatMessage]:
    active_memories = []
    recent_messages = []
    if context is not None:
        active_memories = [
            {
                "id": memory.id,
                "type": memory.memory_type,
                "subject": memory.subject,
                "predicate": memory.predicate,
                "object": memory.object,
                "content": memory.content,
                "status": memory.status,
            }
            for memory in context.active_semantic_memories[:12]
        ]
        recent_messages = [
            {
                "id": message.id,
                "role": message.role,
                "content": message.raw_content[:500],
            }
            for message in context.recent_messages[-8:]
        ]
    prompt = {
        "task": "Extract durable memory candidates from the current turn.",
        "schema": {
            "candidates": [
                {
                    "layer": "episodic|semantic|procedural",
                    "memory_type": "preference|fact|procedure|episode",
                    "content": "natural language atomic memory",
                    "entities": ["entity"],
                    "weak_structure": {
                        "subject": "string|null",
                        "predicate": "string|null",
                        "object": "string|null",
                    },
                    "confidence": "0..1",
                    "stability": "0..1",
                    "salience": "0..1",
                    "source_ids": source_event_ids,
                    "sensitivity_flags": [],
                    "rationale": "short reason",
                }
            ]
        },
        "source_ids": source_event_ids,
        "recent_session_context": recent_messages,
        "active_semantic_memories": active_memories,
        "current_turn": {
            "user": user_message,
            "assistant": assistant_response,
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Return only a JSON object. Include every required field. "
                "Do not include candidates for secrets, credentials, or explicit "
                "do-not-remember requests."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True)},
    ]


def _explicit_do_not_remember(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    patterns = (
        "do not remember",
        "don't remember",
        "dont remember",
        "do not store",
        "don't store",
        "dont store",
        "do not save",
        "don't save",
        "dont save",
    )
    return any(pattern in normalized for pattern in patterns)


def _optional_clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
