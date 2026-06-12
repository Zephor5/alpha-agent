from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from alpha_agent.gateway.adapters import InboundHandler, PlatformAdapter
from alpha_agent.gateway.models import (
    ConversationSource,
    DeliveryResult,
    InboundMessage,
    OutboundMessage,
)
from alpha_agent.gateway.runner import ActiveTurnGuard, GatewayDeliveryError, GatewayRuntimeBridge
from alpha_agent.gateway.session import (
    GatewayDeduplicator,
    GatewaySessionStore,
    SessionMode,
    generate_session_key,
)
from alpha_agent.state.store import StateStore


def _source(
    *,
    chat_id: str = "chat-1",
    chat_type: str = "group",
    user_id: str = "user-1",
    platform_thread_id: str | None = None,
    message_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConversationSource:
    return ConversationSource(
        platform="telegram",
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="Ada",
        platform_thread_id=platform_thread_id,
        message_id=message_id,
        metadata=metadata or {"tenant": "personal"},
    )


def _store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


class _AgentResult:
    def __init__(self, response: str):
        self.response = response


class _FakeAgent:
    def __init__(self, response: str = "runtime response", *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def respond(
        self,
        text: str,
        *,
        session_id: str,
        source_metadata: dict[str, object] | None = None,
    ) -> _AgentResult:
        self.calls.append((text, session_id, source_metadata or {}))
        if self.error:
            raise self.error
        return _AgentResult(self.response)


class _FakeAgentManager:
    def __init__(self, agent: _FakeAgent):
        self.agent = agent
        self.session_ids: list[str] = []

    def get_or_create(self, session_id: str) -> _FakeAgent:
        self.session_ids.append(session_id)
        return self.agent


class _FakeAdapter(PlatformAdapter):
    name = "fake"

    def __init__(
        self,
        messages: list[InboundMessage] | None = None,
        *,
        delivery_success: bool = True,
        fail_start_hook: bool = False,
        fail_complete_hook: bool = False,
    ):
        self.messages = messages or []
        self.delivery_success = delivery_success
        self.fail_start_hook = fail_start_hook
        self.fail_complete_hook = fail_complete_hook
        self.connected = False
        self.disconnected = False
        self.sent: list[tuple[ConversationSource, str]] = []
        self.hooks: list[str] = []

    def connect(self, handler: InboundHandler) -> None:
        self.connected = True
        for message in self.messages:
            handler(message)

    def disconnect(self) -> None:
        self.disconnected = True

    def send(self, source: ConversationSource, outbound: OutboundMessage) -> DeliveryResult:
        self.sent.append((source, outbound.text))
        error = None if self.delivery_success else "denied"
        return DeliveryResult(success=self.delivery_success, error=error)

    def send_typing(self, source: ConversationSource) -> None:
        return None

    def on_processing_start(self, source: ConversationSource) -> None:
        self.hooks.append("start")
        if self.fail_start_hook:
            raise RuntimeError("start hook failed")

    def on_processing_complete(self, source: ConversationSource) -> None:
        self.hooks.append("complete")
        if self.fail_complete_hook:
            raise RuntimeError("complete hook failed")


def test_session_key_generation_modes_are_explicit_and_scoped() -> None:
    source = _source(platform_thread_id="thread-9")

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

    same_group_other_user = _source(user_id="user-2", platform_thread_id="thread-9")
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
    source = _source(platform_thread_id="thread-9")

    first = gateway_sessions.get_or_create(source, SessionMode.THREAD_PER_USER)
    second = gateway_sessions.get_or_create(source, SessionMode.THREAD_PER_USER)

    assert second.session_id == first.session_id
    assert second.session_key == first.session_key
    assert second.source_context["platform"] == "telegram"
    assert second.source_context["session_mode"] == "thread_per_user"
    assert second.source_context["chat_id"] == "chat-1"
    assert second.source_context["user_id"] == "user-1"
    assert second.source_context["platform_thread_id"] == "thread-9"


def test_gateway_session_mapping_creates_session_record_with_explicit_timezone(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    source = _source(metadata={"tenant": "personal", "timezone": "Asia/Shanghai"})

    mapping = gateway_sessions.get_or_create(source, SessionMode.GROUP_PER_USER)

    record = store.get_session_record(mapping.session_id)
    assert record is not None
    assert record.timezone == "Asia/Shanghai"


def test_gateway_session_mapping_uses_local_timezone_fallback_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)

    mapping = gateway_sessions.get_or_create(_source(), SessionMode.GROUP_PER_USER)

    record = store.get_session_record(mapping.session_id)
    assert record is not None
    assert record.timezone == "Asia/Shanghai"


def test_gateway_session_mapping_rejects_invalid_explicit_timezone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    source = _source(metadata={"tenant": "personal", "timezone": "Moon/Base"})

    with pytest.raises(ValueError, match="timezone"):
        gateway_sessions.get_or_create(source, SessionMode.GROUP_PER_USER)


def test_gateway_session_mapping_reuse_ignores_invalid_later_timezone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    first_source = _source(metadata={"tenant": "personal", "timezone": "Asia/Shanghai"})
    later_source = _source(metadata={"tenant": "personal", "timezone": "Moon/Base"})

    first = gateway_sessions.get_or_create(first_source, SessionMode.GROUP_PER_USER)
    second = gateway_sessions.get_or_create(later_source, SessionMode.GROUP_PER_USER)

    record = store.get_session_record(first.session_id)
    assert second.session_id == first.session_id
    assert record is not None
    assert record.timezone == "Asia/Shanghai"


def test_gateway_session_mapping_reuse_ignores_non_string_later_timezone(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    first_source = _source(metadata={"tenant": "personal", "timezone": "Asia/Shanghai"})
    later_source = _source(metadata={"tenant": "personal", "timezone": ["UTC"]})

    first = gateway_sessions.get_or_create(first_source, SessionMode.GROUP_PER_USER)
    second = gateway_sessions.get_or_create(later_source, SessionMode.GROUP_PER_USER)

    record = store.get_session_record(first.session_id)
    assert second.session_id == first.session_id
    assert record is not None
    assert record.timezone == "Asia/Shanghai"


def test_group_shared_session_mapping_keeps_source_context(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)

    mapping = gateway_sessions.get_or_create(_source(user_id="user-1"), SessionMode.GROUP_SHARED)

    assert mapping.source_context["platform"] == "telegram"
    assert mapping.source_context["chat_id"] == "chat-1"
    assert mapping.source_context["user_id"] == "user-1"
    assert mapping.source_context["session_mode"] == "group_shared"


def test_session_mapping_concurrent_creation_reuses_one_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gateway_sessions = GatewaySessionStore(store)
    source = _source(platform_thread_id="thread-9")

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


def test_thread_session_mode_requires_platform_thread_id() -> None:
    source = _source(platform_thread_id=None)

    with pytest.raises(ValueError, match="platform_thread_id is required"):
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


def test_dedup_cached_outbound_ignores_malicious_raw_metadata(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dedup = GatewayDeduplicator(store)
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
        raw_metadata={
            "cached_outbound": {
                "text": "spoofed",
                "delivered": False,
            },
            "gateway_cached_outbound": {
                "text": "spoofed internal",
                "delivered": False,
            },
        },
    )

    result = dedup.check_and_record(message)

    assert dedup.cached_outbound(result.dedup_key) is None


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


def test_gateway_bridge_handles_inbound_turn_and_sends_runtime_response(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _FakeAgent("hello from alpha")
    adapter = _FakeAdapter()
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
    )

    outbound = bridge.handle_message(adapter, message)

    assert outbound is not None
    assert outbound.text == "hello from alpha"
    assert adapter.sent == [(message.source, "hello from alpha")]
    assert adapter.hooks == ["start", "complete"]
    mapping = bridge.session_store.get_or_create(message.source, SessionMode.GROUP_PER_USER)
    assert agent.calls == [
        (
            "hello",
            mapping.session_id,
            {
                "channel": "gateway",
                "platform": "telegram",
                "chat_id": "chat-1",
                "chat_type": "group",
                "user_id": "user-1",
                "user_name": "Ada",
                "message_type": "text",
                "message_id": "msg-1",
                "source": {"tenant": "personal"},
            },
        )
    ]


def test_gateway_bridge_resolves_agent_per_mapped_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _FakeAgent("manager response")
    manager = _FakeAgentManager(agent)
    adapter = _FakeAdapter()
    bridge = GatewayRuntimeBridge(
        agent_manager=manager,
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
    )

    outbound = bridge.handle_message(adapter, message)

    mapping = bridge.session_store.get_or_create(message.source, SessionMode.GROUP_PER_USER)
    assert outbound is not None
    assert outbound.text == "manager response"
    assert manager.session_ids == [mapping.session_id]


def test_gateway_bridge_constructor_requires_agent_manager(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(TypeError, match="agent_manager"):
        GatewayRuntimeBridge(
            session_store=GatewaySessionStore(store),
            deduplicator=GatewayDeduplicator(store),
            turn_guard=ActiveTurnGuard(),
            session_mode=SessionMode.GROUP_PER_USER,
        )  # type: ignore[call-arg]


def test_gateway_bridge_suppresses_duplicate_inbound_without_runtime_or_send(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    agent = _FakeAgent()
    adapter = _FakeAdapter()
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
    )

    first = bridge.handle_message(adapter, message)
    duplicate = bridge.handle_message(adapter, message)

    assert first is not None
    assert duplicate is None
    assert len(agent.calls) == 1
    assert len(adapter.sent) == 1


def test_gateway_bridge_sends_busy_message_when_session_has_active_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = _source(message_id="msg-1")
    session_store = GatewaySessionStore(store)
    mapping = session_store.get_or_create(source, SessionMode.GROUP_PER_USER)
    guard = ActiveTurnGuard()
    guard.begin(mapping.session_id, "already running")
    agent = _FakeAgent()
    adapter = _FakeAdapter()
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=session_store,
        deduplicator=GatewayDeduplicator(store),
        turn_guard=guard,
        session_mode=SessionMode.GROUP_PER_USER,
    )

    outbound = bridge.handle_message(
        adapter,
        InboundMessage(source=source, text="second", platform_message_id="msg-1"),
    )

    assert outbound is not None
    assert "active Alpha turn" in outbound.text
    assert adapter.sent == [(source, outbound.text)]
    assert agent.calls == []
    assert guard.is_active(mapping.session_id) is True


def test_gateway_bridge_delivers_runtime_failure_and_releases_turn_guard(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = _source(message_id="msg-1")
    session_store = GatewaySessionStore(store)
    agent = _FakeAgent(error=RuntimeError("provider failed"))
    adapter = _FakeAdapter()
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=session_store,
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )

    outbound = bridge.handle_message(
        adapter,
        InboundMessage(source=source, text="hello", platform_message_id="msg-1"),
    )
    mapping = session_store.get_or_create(source, SessionMode.GROUP_PER_USER)

    assert outbound is not None
    assert "failed while processing" in outbound.text
    assert adapter.sent == [(source, outbound.text)]
    assert adapter.hooks == ["start", "complete"]
    assert bridge.turn_guard.is_active(mapping.session_id) is False


def test_gateway_bridge_raises_explicit_delivery_error_on_send_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    adapter = _FakeAdapter(delivery_success=False)
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(_FakeAgent()),
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )

    with pytest.raises(GatewayDeliveryError, match="Gateway outbound delivery failed"):
        bridge.handle_message(
            adapter,
            InboundMessage(
                source=_source(message_id="msg-1"),
                text="hello",
                platform_message_id="msg-1",
            ),
        )


def test_gateway_bridge_retries_cached_outbound_without_rerunning_runtime_after_send_failure(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    agent = _FakeAgent()
    adapter = _FakeAdapter(delivery_success=False)
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
    )
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
    )

    with pytest.raises(GatewayDeliveryError):
        bridge.handle_message(adapter, message)
    adapter.delivery_success = True
    retry = bridge.handle_message(adapter, message)

    assert retry is not None
    assert len(agent.calls) == 1
    assert len(adapter.sent) == 2
    assert adapter.sent == [
        (message.source, "runtime response"),
        (message.source, "runtime response"),
    ]


def test_gateway_bridge_ignores_start_hook_failure_and_still_sends(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    agent = _FakeAgent("runtime response")
    adapter = _FakeAdapter(fail_start_hook=True)
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=GatewaySessionStore(store),
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
        error_log_path=tmp_path / "errors.log",
    )
    message = InboundMessage(
        source=_source(message_id="msg-1"),
        text="hello",
        platform_message_id="msg-1",
    )

    outbound = bridge.handle_message(adapter, message)

    assert outbound is not None
    assert outbound.text == "runtime response"
    assert len(agent.calls) == 1
    assert adapter.sent == [(message.source, "runtime response")]
    assert "gateway.adapter_hook.error" in (tmp_path / "errors.log").read_text(encoding="utf-8")


def test_gateway_bridge_ignores_complete_hook_failure_and_releases_guard(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source = _source(message_id="msg-1")
    session_store = GatewaySessionStore(store)
    agent = _FakeAgent("runtime response")
    adapter = _FakeAdapter(fail_complete_hook=True)
    bridge = GatewayRuntimeBridge(
        agent_manager=_FakeAgentManager(agent),
        session_store=session_store,
        deduplicator=GatewayDeduplicator(store),
        turn_guard=ActiveTurnGuard(),
        session_mode=SessionMode.GROUP_PER_USER,
        error_log_path=tmp_path / "errors.log",
    )

    outbound = bridge.handle_message(
        adapter,
        InboundMessage(source=source, text="hello", platform_message_id="msg-1"),
    )
    mapping = session_store.get_or_create(source, SessionMode.GROUP_PER_USER)

    assert outbound is not None
    assert outbound.text == "runtime response"
    assert adapter.sent == [(source, "runtime response")]
    assert bridge.turn_guard.is_active(mapping.session_id) is False
    assert "gateway.adapter_hook.error" in (tmp_path / "errors.log").read_text(encoding="utf-8")
