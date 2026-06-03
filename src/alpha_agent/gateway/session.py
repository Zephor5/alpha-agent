"""Gateway session mapping and inbound deduplication services."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from alpha_agent.gateway.models import (
    ConversationSource,
    InboundMessage,
    OutboundMessage,
    Visibility,
)
from alpha_agent.runtime.session import new_session_id
from alpha_agent.state.store import StateStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now

GATEWAY_CACHED_OUTBOUND_KEY = "gateway_cached_outbound"
RAW_METADATA_KEY = "raw_metadata"


class SessionMode(StrEnum):
    """Supported mappings from external conversations to Alpha sessions."""

    DM = "dm"
    GROUP_SHARED = "group_shared"
    GROUP_PER_USER = "group_per_user"
    THREAD = "thread"
    THREAD_PER_USER = "thread_per_user"


@dataclass(frozen=True, slots=True)
class GatewaySessionMapping:
    """Persisted mapping from an external conversation scope to an Alpha session."""

    id: str
    session_key: str
    session_id: str
    session_mode: SessionMode
    source_context: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Result of recording or rejecting an inbound message deduplication key."""

    duplicate: bool
    dedup_key: str
    expires_at: str | None = None


def generate_session_key(source: ConversationSource, mode: SessionMode) -> str:
    """Generate a stable, explicit key for an external conversation session mode."""

    components = _session_components(source, mode)
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return f"gateway:v1:{mode.value}:{payload}"


class GatewaySessionStore:
    """Persistence for external gateway conversation scopes and Alpha session ids."""

    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    def get_or_create(
        self,
        source: ConversationSource,
        mode: SessionMode,
    ) -> GatewaySessionMapping:
        """Return the existing Alpha session mapping for source and mode, or create one."""

        session_key = generate_session_key(source, mode)
        with self.state_store.transaction() as conn:
            now = _iso(utc_now())
            source_context = _source_context(source, mode, session_key)
            mapping_id = new_id("gateway_session")
            session_id = new_session_id()
            conn.execute(
                """
                INSERT OR IGNORE INTO gateway_session_mappings
                    (id, platform, chat_id, chat_type, user_id, platform_thread_id, session_mode,
                     session_key, session_id, source_context, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mapping_id,
                    _norm_required(source.platform, "platform").lower(),
                    _norm_required(source.chat_id, "chat_id"),
                    source.chat_type.strip().lower(),
                    _norm_required(source.user_id, "user_id"),
                    _norm_optional(source.platform_thread_id),
                    mode.value,
                    session_key,
                    session_id,
                    _dumps(source_context),
                    now,
                    now,
                    _dumps(source.metadata),
                ),
            )
            row = self._find_by_key(conn, session_key)
            if row is None:
                raise RuntimeError("gateway session mapping insert did not return a row")
            return self._mapping_from_row(row)

    def _find_by_key(self, conn: sqlite3.Connection, session_key: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM gateway_session_mappings WHERE session_key = ?",
            (session_key,),
        ).fetchone()

    def _mapping_from_row(self, row: sqlite3.Row) -> GatewaySessionMapping:
        return GatewaySessionMapping(
            id=row["id"],
            session_key=row["session_key"],
            session_id=row["session_id"],
            session_mode=SessionMode(row["session_mode"]),
            source_context=_loads_dict(row["source_context"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class GatewayDeduplicator:
    """Durable inbound deduplication keyed by platform ids or short-lived text fingerprints."""

    def __init__(
        self,
        state_store: StateStore,
        fallback_ttl: timedelta = timedelta(minutes=2),
    ):
        self.state_store = state_store
        self.fallback_ttl = fallback_ttl

    def check_and_record(
        self,
        message: InboundMessage,
        *,
        now: datetime | None = None,
    ) -> DedupResult:
        """Return duplicate=True when message was already recorded."""

        current = now or utc_now()
        platform_message_id = _norm_optional(
            message.platform_message_id or message.source.message_id,
        )
        if platform_message_id:
            dedup_key = _platform_message_key(message.source, platform_message_id)
            expires_at = None
            fingerprint = None
        else:
            fingerprint = _fallback_fingerprint(message)
            dedup_key = f"fallback:{fingerprint}"
            expires_at = _iso(current + self.fallback_ttl)

        with self.state_store.transaction() as conn:
            self._prune_expired_fallbacks(conn, current)
            try:
                conn.execute(
                    """
                    INSERT INTO gateway_dedup
                        (id, dedup_key, platform, chat_id, platform_message_id, fingerprint,
                         created_at, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id("gateway_dedup"),
                        dedup_key,
                        _norm_required(message.source.platform, "platform").lower(),
                        _norm_required(message.source.chat_id, "chat_id"),
                        platform_message_id,
                        fingerprint,
                        _iso(current),
                        expires_at,
                        _dumps({RAW_METADATA_KEY: message.raw_metadata}),
                    ),
                )
            except sqlite3.IntegrityError:
                return DedupResult(duplicate=True, dedup_key=dedup_key, expires_at=expires_at)
        return DedupResult(duplicate=False, dedup_key=dedup_key, expires_at=expires_at)

    def cache_outbound(self, dedup_key: str, outbound: OutboundMessage) -> None:
        """Persist outbound data for retrying delivery without rerunning Alpha."""

        with self.state_store.transaction() as conn:
            row = conn.execute(
                "SELECT metadata FROM gateway_dedup WHERE dedup_key = ?",
                (dedup_key,),
            ).fetchone()
            if row is None:
                return
            metadata = _loads_dict(row["metadata"])
            metadata[GATEWAY_CACHED_OUTBOUND_KEY] = _outbound_to_metadata(outbound)
            conn.execute(
                "UPDATE gateway_dedup SET metadata = ? WHERE dedup_key = ?",
                (_dumps(metadata), dedup_key),
            )

    def mark_outbound_delivered(self, dedup_key: str) -> None:
        """Mark cached outbound as delivered so later duplicates remain suppressed."""

        with self.state_store.transaction() as conn:
            row = conn.execute(
                "SELECT metadata FROM gateway_dedup WHERE dedup_key = ?",
                (dedup_key,),
            ).fetchone()
            if row is None:
                return
            metadata = _loads_dict(row["metadata"])
            cached = metadata.get(GATEWAY_CACHED_OUTBOUND_KEY)
            if not isinstance(cached, dict):
                return
            cached["delivered"] = True
            metadata[GATEWAY_CACHED_OUTBOUND_KEY] = cached
            conn.execute(
                "UPDATE gateway_dedup SET metadata = ? WHERE dedup_key = ?",
                (_dumps(metadata), dedup_key),
            )

    def cached_outbound(self, dedup_key: str) -> OutboundMessage | None:
        """Return cached outbound that still needs delivery retry."""

        with self.state_store.connect() as conn:
            row = conn.execute(
                "SELECT metadata FROM gateway_dedup WHERE dedup_key = ?",
                (dedup_key,),
            ).fetchone()
        if row is None:
            return None
        metadata = _loads_dict(row["metadata"])
        cached = metadata.get(GATEWAY_CACHED_OUTBOUND_KEY)
        if not isinstance(cached, dict):
            return None
        if cached.get("delivered") is True:
            return None
        return _outbound_from_metadata(cached)

    def _prune_expired_fallbacks(self, conn: sqlite3.Connection, now: datetime) -> None:
        conn.execute(
            """
            DELETE FROM gateway_dedup
            WHERE expires_at IS NOT NULL AND expires_at <= ?
            """,
            (_iso(now),),
        )


def _session_components(source: ConversationSource, mode: SessionMode) -> dict[str, str]:
    platform = _norm_required(source.platform, "platform").lower()
    chat_id = _norm_required(source.chat_id, "chat_id")
    user_id = _norm_required(source.user_id, "user_id")
    platform_thread_id = _norm_optional(source.platform_thread_id)

    if mode == SessionMode.DM:
        return {"platform": platform, "user_id": user_id}
    if mode == SessionMode.GROUP_SHARED:
        return {"platform": platform, "chat_id": chat_id}
    if mode == SessionMode.GROUP_PER_USER:
        return {"platform": platform, "chat_id": chat_id, "user_id": user_id}
    if mode == SessionMode.THREAD:
        if platform_thread_id is None:
            raise ValueError("platform_thread_id is required for thread session mode")
        return {
            "platform": platform,
            "chat_id": chat_id,
            "platform_thread_id": platform_thread_id,
        }
    if mode == SessionMode.THREAD_PER_USER:
        if platform_thread_id is None:
            raise ValueError("platform_thread_id is required for thread_per_user session mode")
        return {
            "platform": platform,
            "chat_id": chat_id,
            "platform_thread_id": platform_thread_id,
            "user_id": user_id,
        }
    raise ValueError(f"unsupported session mode: {mode}")


def _source_context(
    source: ConversationSource,
    mode: SessionMode,
    session_key: str,
) -> dict[str, Any]:
    platform = _norm_required(source.platform, "platform").lower()
    chat_id = _norm_required(source.chat_id, "chat_id")
    user_id = _norm_required(source.user_id, "user_id")
    platform_thread_id = _norm_optional(source.platform_thread_id)
    return {
        "platform": platform,
        "chat_id": chat_id,
        "chat_type": source.chat_type.strip().lower(),
        "user_id": user_id,
        "user_name": source.user_name,
        "platform_thread_id": platform_thread_id,
        "session_mode": mode.value,
        "session_key": session_key,
        "external_metadata": dict(source.metadata),
    }


def _platform_message_key(source: ConversationSource, platform_message_id: str) -> str:
    payload = {
        "platform": _norm_required(source.platform, "platform").lower(),
        "chat_id": _norm_required(source.chat_id, "chat_id"),
        "message_id": platform_message_id,
    }
    return "platform_message:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fallback_fingerprint(message: InboundMessage) -> str:
    source = message.source
    payload = {
        "platform": _norm_required(source.platform, "platform").lower(),
        "chat_id": _norm_required(source.chat_id, "chat_id"),
        "platform_thread_id": _norm_optional(source.platform_thread_id),
        "user_id": _norm_required(source.user_id, "user_id"),
        "message_type": message.message_type.strip().lower(),
        "text": " ".join(message.text.split()).casefold(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _outbound_to_metadata(outbound: OutboundMessage) -> dict[str, Any]:
    return {
        "text": outbound.text,
        "attachments": outbound.attachments,
        "reply_to": outbound.reply_to,
        "thread_metadata": outbound.thread_metadata,
        "visibility": outbound.visibility,
        "delivered": False,
    }


def _outbound_from_metadata(value: dict[str, Any]) -> OutboundMessage | None:
    text = value.get("text")
    if not isinstance(text, str):
        return None
    attachments = value.get("attachments", [])
    thread_metadata = value.get("thread_metadata", {})
    visibility = value.get("visibility", "default")
    reply_to = value.get("reply_to")
    if not isinstance(attachments, list):
        attachments = []
    if not isinstance(thread_metadata, dict):
        thread_metadata = {}
    if not isinstance(visibility, str):
        visibility = "default"
    if visibility not in {"default", "public", "ephemeral"}:
        visibility = "default"
    typed_visibility = cast(Visibility, visibility)
    if reply_to is not None and not isinstance(reply_to, str):
        reply_to = None
    return OutboundMessage(
        text=text,
        attachments=attachments,
        reply_to=reply_to,
        thread_metadata=thread_metadata,
        visibility=typed_visibility,
    )


def _norm_required(value: str | None, field_name: str) -> str:
    normalized = value.strip() if value else ""
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _norm_optional(value: str | None) -> str | None:
    normalized = value.strip() if value else ""
    return normalized or None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}
