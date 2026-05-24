"""Policy boundary for memory scope, candidates, promotion, and retrieval."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from typing import Any

from alpha_agent.memory.extractor import MemoryExtractionContext, MemoryExtractor
from alpha_agent.memory.models import (
    ExtractedMemoryCandidate,
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    RetrievedContext,
    SemanticMemory,
    proposed_layer_for_candidate,
)
from alpha_agent.memory.persistence import PersistedMemory, persist_candidates
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class MemoryPromotionPolicyError(ValueError):
    """Raised when a stored candidate is blocked before durable promotion."""

    def __init__(self, candidate_id: str, reason: str):
        super().__init__(f"memory candidate {candidate_id} blocked by policy: {reason}")
        self.candidate_id = candidate_id
        self.reason = reason


class MemoryController:
    """Own memory policy decisions outside the runtime orchestration body."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        retriever: MemoryRetriever,
        extractor: MemoryExtractor | None = None,
    ):
        self.store = store
        self.retriever = retriever
        self.extractor = extractor or MemoryExtractor()

    def scope_for_turn(
        self,
        *,
        session_id: str,
        source_metadata: dict[str, Any] | None,
    ) -> MemoryScope:
        """Derive the current write scope from caller source metadata."""

        return MemoryScope.from_source_metadata(
            session_id=session_id,
            source_metadata=source_metadata,
        )

    def retrieve_context(
        self,
        *,
        query: str,
        session_id: str,
        scope: MemoryScope,
        limit: int,
        record_access: bool = True,
    ) -> RetrievedContext:
        """Retrieve only memories visible to the current scope."""

        return self.retriever.retrieve_context(
            query,
            session_id,
            limit=limit,
            scopes=scope.allowed_read_scopes(),
            record_access=record_access,
            access_scope=scope,
        )

    def apply_forget_request(
        self,
        *,
        session_id: str,
        user_message: str,
        context: RetrievedContext,
        scope: MemoryScope,
    ) -> tuple[RetrievedContext, list[str], list[str]]:
        """Apply explicit forget commands and remove forgotten memories from context."""

        forget_ids = self._forget_targets(user_message, context.semantic_memories)
        if not forget_ids:
            return context, [], []
        forgotten: list[str] = []
        skipped: list[str] = []
        visible_ids = {memory.id for memory in context.semantic_memories}
        allowed_scope_keys = {item.scope_key for item in scope.allowed_read_scopes()}
        for memory_id in forget_ids:
            memory = self.store.get_semantic_memory(memory_id)
            if memory is None or (
                memory_id not in visible_ids and memory.scope.scope_key not in allowed_scope_keys
            ):
                skipped.append(memory_id)
                continue
            self.store.forget_semantic_memory(memory_id, reason=user_message)
            forgotten.append(memory_id)
        if skipped:
            self.store.append_runtime_trace(
                session_id=session_id,
                event_type="memory.forget.skipped",
                content=str(len(skipped)),
                metadata={
                    "memory_ids": skipped,
                    "scope": scope.to_record(),
                    "reason": "memory id is outside visible memory scope",
                },
            )
        filtered = RetrievedContext(
            episodic_memories=list(context.episodic_memories),
            semantic_memories=[
                memory for memory in context.semantic_memories if memory.id not in forgotten
            ],
            procedural_memories=list(context.procedural_memories),
            entity_hints=list(context.entity_hints),
        )
        return filtered, forgotten, skipped

    def extract_candidates(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        source_message_ids: list[str],
        scope: MemoryScope,
        retrieved_context: RetrievedContext | None = None,
        recent_messages: list[Any] | None = None,
    ) -> list[MemoryCandidate]:
        """Extract and store candidates before any durable promotion decision."""

        extracted = self.preview_extracted_candidates(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            source_message_ids=source_message_ids,
            scope=scope,
            retrieved_context=retrieved_context,
            recent_messages=recent_messages,
            record_policy_trace=True,
        )
        now = utc_now_iso()
        stored: list[MemoryCandidate] = []
        for candidate in extracted:
            memory_candidate = MemoryCandidate(
                id=new_id("cand"),
                candidate_type=candidate.type,
                proposed_layer=proposed_layer_for_candidate(candidate.type),
                content=candidate.content,
                weak_structure=_candidate_weak_structure(candidate),
                salience=candidate.salience,
                confidence=candidate.confidence,
                scope=scope,
                source_message_ids=list(candidate.source_event_ids),
                status="pending",
                created_at=now,
                updated_at=now,
                metadata=dict(candidate.metadata),
            )
            stored.append(self.store.insert_memory_candidate(memory_candidate))
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.candidates.created",
            content=str(len(stored)),
            metadata={
                "candidate_ids": [candidate.id for candidate in stored],
                "scope": scope.to_record(),
            },
        )
        return stored

    def preview_extracted_candidates(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        source_message_ids: list[str],
        scope: MemoryScope,
        retrieved_context: RetrievedContext | None = None,
        recent_messages: list[Any] | None = None,
        record_policy_trace: bool = False,
    ) -> list[ExtractedMemoryCandidate]:
        """Extract candidates after applying the shared extraction policy gate."""

        source_messages = self.store.list_conversation_messages_by_ids(source_message_ids)
        policy_denial = self.extraction_policy_denial(
            user_message=user_message,
            source_message_ids=source_message_ids,
            scope=scope,
            source_messages=source_messages,
        )
        if policy_denial is not None:
            if record_policy_trace:
                self._trace_extraction_skip(
                    session_id=session_id,
                    reason=policy_denial,
                    source_message_ids=source_message_ids,
                    scope=scope,
                )
            return []

        extraction_context = MemoryExtractionContext(
            session_id=session_id,
            recent_messages=list(recent_messages or source_messages),
            active_semantic_memories=(
                list(retrieved_context.semantic_memories) if retrieved_context is not None else []
            ),
        )
        extracted = self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
            source_event_ids=source_message_ids,
            context=extraction_context,
        )
        allowed: list[ExtractedMemoryCandidate] = []
        for candidate in extracted:
            candidate_denial = self.extracted_candidate_policy_denial(candidate)
            if candidate_denial is not None:
                if record_policy_trace:
                    self.store.append_runtime_trace(
                        session_id=session_id,
                        event_type="memory.extraction.skipped_candidate",
                        content=candidate_denial,
                        metadata={
                            "scope": scope.to_record(),
                            "sensitivity_flags": _sensitivity_flags(candidate),
                        },
                    )
                continue
            allowed.append(candidate)
        return allowed

    def extraction_policy_denial(
        self,
        *,
        user_message: str,
        source_message_ids: list[str],
        scope: MemoryScope,
        source_messages: list[Any] | None = None,
    ) -> str | None:
        """Return the shared request-level extraction policy denial reason."""

        policy_denial = _extraction_policy_denial(user_message, scope=scope)
        if policy_denial is not None:
            return policy_denial
        messages = source_messages
        if messages is None:
            messages = self.store.list_conversation_messages_by_ids(source_message_ids)
        return _source_policy_denial(messages)

    def extracted_candidate_policy_denial(
        self,
        candidate: ExtractedMemoryCandidate,
    ) -> str | None:
        """Return the shared candidate-level extraction policy denial reason."""

        if _candidate_is_sensitive(candidate):
            flags = _sensitivity_flags(candidate) or _detect_sensitivity(candidate.content)
            return "sensitive candidate blocked: " + ",".join(flags)
        return None

    def candidate_promotion_policy_denial(
        self,
        candidate: MemoryCandidate,
    ) -> str | None:
        """Return why a stored candidate cannot be durably promoted, if blocked."""

        source_messages = self.store.list_conversation_messages_by_ids(
            candidate.source_message_ids
        )
        source_policy_denial = _stored_source_policy_denial(
            source_messages=source_messages,
            fallback_texts=[candidate.content],
            scope=candidate.scope,
        )
        if source_policy_denial is not None:
            return source_policy_denial
        return self.extracted_candidate_policy_denial(_stored_candidate_to_extracted(candidate))

    def _trace_extraction_skip(
        self,
        *,
        session_id: str,
        reason: str,
        source_message_ids: list[str],
        scope: MemoryScope,
    ) -> None:
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.extraction.skipped",
            content=reason,
            metadata={
                "scope": scope.to_record(),
                "source_message_ids": source_message_ids,
            },
        )

    def decide_runtime_candidates(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        trusted_scope: bool = True,
        auto_approve_explicit: bool = True,
    ) -> list[PersistedMemory]:
        """Apply runtime auto-approval policy and persist approved candidates."""

        persisted: list[PersistedMemory] = []
        explicit_batch = any(
            candidate.metadata.get("extractor") == "explicit_or_correction"
            for candidate in candidates
        )
        for candidate in candidates:
            if (
                trusted_scope
                and auto_approve_explicit
                and self._should_auto_approve(candidate, explicit_batch=explicit_batch)
            ):
                with self.store.immediate_transaction() as conn:
                    approved = self.store.update_memory_candidate_status(
                        candidate.id,
                        "auto_approved",
                        reviewer_metadata={"reviewer": "memory_controller"},
                        conn=conn,
                    )
                    self.store.insert_memory_decision(
                        MemoryDecision(
                            id=new_id("decision"),
                            candidate_id=approved.id,
                            action="auto_approve",
                            memory_type=None,
                            memory_id=None,
                            reviewer="memory_controller",
                            rationale="explicit high-confidence runtime memory",
                            created_at=utc_now_iso(),
                            metadata={},
                        ),
                        conn=conn,
                    )
                    persisted.extend(
                        self.promote_candidate(
                            approved,
                            action="promote",
                            reviewer="memory_controller",
                            rationale="promoted after runtime auto-approval",
                            conn=conn,
                        )
                    )
                continue
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action="pending",
                    memory_type=None,
                    memory_id=None,
                    reviewer="memory_controller",
                    rationale="candidate requires review",
                    created_at=utc_now_iso(),
                    metadata={},
                )
            )
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.decisions",
            content=str(len(persisted)),
            metadata={
                "persisted": [item.__dict__ for item in persisted],
                "candidate_count": len(candidates),
            },
        )
        return persisted

    def promote_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        action: str,
        reviewer: str | None,
        rationale: str,
        conn: sqlite3.Connection | None = None,
    ) -> list[PersistedMemory]:
        """Persist one approved candidate and record the decision."""

        policy_denial = self.candidate_promotion_policy_denial(candidate)
        if policy_denial is not None:
            raise MemoryPromotionPolicyError(candidate.id, policy_denial)
        if conn is not None:
            return self._promote_candidate_in_conn(
                candidate,
                action=action,
                reviewer=reviewer,
                rationale=rationale,
                conn=conn,
            )
        with self.store.immediate_transaction() as local:
            return self._promote_candidate_in_conn(
                candidate,
                action=action,
                reviewer=reviewer,
                rationale=rationale,
                conn=local,
            )

    def _promote_candidate_in_conn(
        self,
        candidate: MemoryCandidate,
        *,
        action: str,
        reviewer: str | None,
        rationale: str,
        conn: sqlite3.Connection,
    ) -> list[PersistedMemory]:
        extracted = _stored_candidate_to_extracted(candidate)
        persisted = persist_candidates(
            self.store,
            [extracted],
            scope=candidate.scope,
            conn=conn,
        )
        if not persisted:
            raise ValueError(f"memory candidate {candidate.id} cannot be promoted")
        for item in persisted:
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action=item.action,
                    memory_type=item.memory_type,
                    memory_id=item.memory_id,
                    reviewer=reviewer,
                    rationale=rationale,
                    created_at=utc_now_iso(),
                    metadata={
                        "candidate_type": item.candidate_type,
                        "requested_action": action,
                    },
                ),
                conn=conn,
            )
        return persisted

    def _should_auto_approve(
        self,
        candidate: MemoryCandidate,
        *,
        explicit_batch: bool = False,
    ) -> bool:
        if explicit_batch and candidate.confidence >= 0.65:
            return True
        extractor = str(candidate.metadata.get("extractor") or "")
        if extractor == "explicit_or_correction":
            return candidate.confidence >= 0.7
        source_text = candidate.content.casefold()
        explicit = "remember" in source_text or "user said:" in source_text
        return explicit and candidate.confidence >= 0.65 and candidate.salience >= 0.75

    def _forget_targets(
        self,
        user_message: str,
        semantic_memories: list[SemanticMemory],
    ) -> list[str]:
        normalized = " ".join(user_message.casefold().split())
        if not normalized.startswith("forget"):
            return []
        explicit_ids = re.findall(r"\bsem_[a-zA-Z0-9_-]+\b", user_message)
        if explicit_ids:
            return list(dict.fromkeys(explicit_ids))
        if normalized in {"forget this", "forget that"} and semantic_memories:
            return [semantic_memories[0].id]
        if normalized.startswith("forget memory "):
            candidate_id = user_message.split("forget memory ", 1)[-1].strip()
            return [candidate_id] if candidate_id else []
        return []


def _candidate_weak_structure(candidate: ExtractedMemoryCandidate) -> dict[str, Any]:
    weak = {
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "object": candidate.object,
    }
    if candidate.entities:
        weak["entities"] = list(candidate.entities)
    return weak


def _stored_candidate_to_extracted(candidate: MemoryCandidate) -> ExtractedMemoryCandidate:
    weak = candidate.weak_structure
    entities = weak.get("entities")
    metadata = dict(candidate.metadata)
    return ExtractedMemoryCandidate(
        type=candidate.candidate_type,
        content=candidate.content,
        salience=candidate.salience,
        confidence=candidate.confidence,
        stability=_float_metadata(metadata.get("stability"), 0.6),
        entities=[str(item) for item in entities] if isinstance(entities, list) else [],
        subject=_optional_str(weak.get("subject")),
        predicate=_optional_str(weak.get("predicate")),
        object=_optional_str(weak.get("object")),
        source_event_ids=list(candidate.source_message_ids),
        metadata=metadata,
    )


def edited_candidate(
    candidate: ExtractedMemoryCandidate,
    *,
    source_message_ids: list[str],
) -> ExtractedMemoryCandidate:
    """Return a candidate with stored source ids preserved after review edits."""

    return replace(candidate, source_event_ids=source_message_ids)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extraction_policy_denial(user_message: str, *, scope: MemoryScope) -> str | None:
    if _explicit_do_not_remember(user_message):
        return "explicit do-not-remember request"
    sensitivity = _detect_sensitivity(user_message)
    if sensitivity:
        return "sensitive data blocked: " + ",".join(sensitivity)
    if _is_group_scope(scope) and not _explicit_remember(user_message):
        return "group chat ambient memory write blocked"
    return None


def _source_policy_denial(source_messages: list[Any]) -> str | None:
    return _stored_source_policy_denial(
        source_messages=source_messages,
        fallback_texts=[],
        scope=None,
    )


def _stored_source_policy_denial(
    *,
    source_messages: list[Any],
    fallback_texts: list[str],
    scope: MemoryScope | None,
) -> str | None:
    source_texts = [
        str(getattr(message, "raw_content", "") or "")
        for message in source_messages
        if str(getattr(message, "raw_content", "") or "").strip()
    ]
    texts = [*source_texts, *fallback_texts]
    for message in source_messages:
        metadata = getattr(message, "source_metadata", {}) or {}
        if _is_platform_system_metadata(metadata):
            return "platform/system source message blocked"
    for text in texts:
        if _explicit_do_not_remember(text):
            return "explicit do-not-remember request"
    for text in texts:
        sensitivity = _detect_sensitivity(text)
        if sensitivity:
            return "sensitive data blocked: " + ",".join(sensitivity)
    if scope is not None and _is_group_scope(scope):
        explicit = any(_explicit_remember(text) for text in texts)
        if not explicit:
            return "group chat ambient memory write blocked"
    return None


def _explicit_do_not_remember(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return any(
        pattern in normalized
        for pattern in (
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
    )


def _explicit_remember(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return "remember that" in normalized or normalized.startswith("remember ")


def _detect_sensitivity(value: str) -> list[str]:
    checks = {
        "password": r"\b(password|passcode|pin)\b\s*(?:is|=|:)",
        "api_key": r"\b(api[_ -]?key|secret[_ -]?key)\b\s*(?:is|=|:)",
        "token": r"\b(access[_ -]?token|refresh[_ -]?token|bearer token)\b\s*(?:is|=|:)",
        "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,19}\b",
    }
    return [
        name
        for name, pattern in checks.items()
        if re.search(pattern, value, flags=re.IGNORECASE)
    ]


def _candidate_is_sensitive(candidate: ExtractedMemoryCandidate) -> bool:
    return bool(_sensitivity_flags(candidate) or _detect_sensitivity(candidate.content))


def _sensitivity_flags(candidate: ExtractedMemoryCandidate) -> list[str]:
    flags = candidate.metadata.get("sensitivity_flags")
    if not isinstance(flags, list):
        return []
    return [str(flag) for flag in flags if str(flag).strip()]


def _is_group_scope(scope: MemoryScope) -> bool:
    chat_type = str(scope.metadata.get("chat_type") or "").strip().casefold()
    return scope.kind == "chat_thread" and chat_type not in {"", "dm", "direct", "private"}


def _is_platform_system_metadata(metadata: dict[str, Any]) -> bool:
    if metadata.get("is_system_message") is True:
        return True
    for key in ("message_type", "sender_type", "event_type", "subtype"):
        value = str(metadata.get(key) or "").strip().casefold()
        if value in {"system", "platform", "bot", "control", "event"}:
            return True
    return False


def _float_metadata(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
