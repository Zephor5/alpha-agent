from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alpha_agent.gateway.models import ConversationSource, InboundMessage
from alpha_agent.gateway.runner import ActiveTurnGuard
from alpha_agent.gateway.session import (
    GatewayDeduplicator,
    GatewaySessionStore,
    SessionMode,
    generate_session_key,
)
from alpha_agent.memory.store import MemoryStore


def _source(
    *,
    chat_id: str = "chat-1",
    chat_type: str = "group",
    user_id: str = "user-1",
    thread_id: str | None = None,
    message_id: str | None = None,
) -> ConversationSource:
    return ConversationSource(
        platform="telegram",
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="Ada",
        thread_id=thread_id,
        message_id=message_id,
        metadata={"tenant": "personal"},
    )


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def test_session_key_generation_modes_are_explicit_and_scoped() -> None:
    source = _source(thread_id="thread-9")

    dm = generate_session_key(source, SessionMode.DM)
    group_shared = generate_session_key(source, SessionMode.GROUP_SHARED)
    group_per_user = generate_session_key(source, SessionMode.GROUP_PER_USER)
    thread = generate_session_key(source, SessionMode.THREAD)
    thread_per_user = generate_session_key(source, SessionMode.THREAD_PER_USER)

    assert "dm" in dm
    assert "group_shared" in group_shared
    assert "group_per_user" in group_per_user
    assert "thread" in thread
    assert "thread_per_user" in thread_per_user
    assert len({dm, group_shared, group_per_user, thread, thread_per_user}) == 5

    same_group_other_user = _source(user_id="user-2", thread_id="thread-9")
    assert generate_session_key(source, SessionMode.GROUP_SHARED) == generate_session_key(
        same_group_other_user,
        SessionMode.GROUP_SHARED,
    )
    assert generate_session_key(source, SessionMode.GROUP_PER_USER) != generate_session_key(
        same_group_other_user,
        SessionMode.GROUP_PER_USER,
    )


def test_session_mapping_is_persisted_and_reused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    source = _source(thread_id="thread-9")

    first = gateway_sessions.get_or_create(source, SessionMode.THREAD_PER_USER)
    second = gateway_sessions.get_or_create(source, SessionMode.THREAD_PER_USER)

    assert second.session_id == first.session_id
    assert second.session_key == first.session_key
    assert second.memory_scope["platform"] == "telegram"
    assert second.memory_scope["session_mode"] == "thread_per_user"
    assert second.memory_scope["chat_id"] == "chat-1"
    assert second.memory_scope["user_id"] == "user-1"
    assert second.memory_scope["thread_id"] == "thread-9"


def test_session_mapping_concurrent_creation_reuses_one_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    source = _source(thread_id="thread-9")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: gateway_sessions.get_or_create(source, SessionMode.THREAD_PER_USER),
                range(16),
            )
        )

    assert len({result.session_id for result in results}) == 1
    assert len({result.session_key for result in results}) == 1

    with store.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM gateway_session_mappings").fetchone()
    assert row["count"] == 1


def test_thread_session_mode_requires_thread_id() -> None:
    source = _source(thread_id=None)

    with pytest.raises(ValueError, match="thread_id is required"):
        generate_session_key(source, SessionMode.THREAD)


def test_dedup_by_platform_message_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dedup = GatewayDeduplicator(store)
    source = _source(message_id="platform-msg-1")
    message = InboundMessage(
        source=source,
        text="hello",
        platform_message_id="platform-msg-1",
    )

    first = dedup.check_and_record(message)
    second = dedup.check_and_record(message)

    assert first.duplicate is False
    assert second.duplicate is True


def test_dedup_by_fallback_text_fingerprint_honors_ttl(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dedup = GatewayDeduplicator(store, fallback_ttl=timedelta(seconds=30))
    source = _source()
    message = InboundMessage(source=source, text="Hello   gateway")
    start = datetime(2026, 1, 1, tzinfo=UTC)

    first = dedup.check_and_record(message, now=start)
    duplicate = dedup.check_and_record(message, now=start + timedelta(seconds=10))
    after_ttl = dedup.check_and_record(message, now=start + timedelta(seconds=31))

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert after_ttl.duplicate is False


def test_fallback_dedup_prunes_expired_row_before_reinsert(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dedup = GatewayDeduplicator(store, fallback_ttl=timedelta(seconds=30))
    message = InboundMessage(source=_source(), text="Hello gateway")
    start = datetime(2026, 1, 1, tzinfo=UTC)

    first = dedup.check_and_record(message, now=start)
    after_ttl = dedup.check_and_record(message, now=start + timedelta(seconds=31))

    assert first.duplicate is False
    assert after_ttl.duplicate is False

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT created_at FROM gateway_dedup WHERE dedup_key = ?",
            (first.dedup_key,),
        ).fetchall()
    assert [row["created_at"] for row in rows] == [
        "2026-01-01T00:00:31+00:00",
    ]


def test_active_turn_guard_allows_command_bypass() -> None:
    guard = ActiveTurnGuard()

    first = guard.begin("session-1", "normal work")
    blocked = guard.begin("session-1", "second turn")
    status = guard.begin("session-1", "/status")
    stop = guard.begin("session-1", "/stop now")
    reset = guard.begin("session-1", "/reset")

    assert first.accepted is True
    assert blocked.accepted is False
    assert blocked.reason == "active_turn"
    assert status.accepted is True
    assert status.bypassed is True
    assert stop.accepted is True
    assert stop.bypassed is True
    assert reset.accepted is True
    assert reset.bypassed is True

    guard.complete("session-1")
    assert guard.begin("session-1", "after complete").accepted is True


def test_active_turn_guard_admits_only_one_threaded_begin() -> None:
    guard = ActiveTurnGuard()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: guard.begin("session-1", "work"), range(16)))

    accepted = [result for result in results if result.accepted]
    blocked = [result for result in results if not result.accepted]

    assert len(accepted) == 1
    assert all(result.reason == "active_turn" for result in blocked)
    assert guard.is_active("session-1") is True
