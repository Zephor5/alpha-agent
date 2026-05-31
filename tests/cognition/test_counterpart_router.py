from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    Stimulus,
    StimulusKind,
    ThreadId,
)
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.runtime.counterpart_router import DEFAULT_COUNTERPART_ID, CounterpartRouter
from alpha_agent.state.store import StateStore
from alpha_agent.tools.default import build_tool_registry


def test_counterpart_router_maps_local_and_first_channel_user_to_default() -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log)
    router = CounterpartRouter(log)

    local = router.upsert_from_source_metadata(
        {"channel": "cli", "command": "ask"},
        emitter=emitter,
    )
    first_channel_user = router.upsert_from_source_metadata(
        {
            "channel": "gateway",
            "platform": "slack",
            "user_id": "u1",
            "user_name": "Eric",
        },
        emitter=emitter,
    )

    assert local is not None
    assert local.id == str(DEFAULT_COUNTERPART_ID)
    assert first_channel_user == local


def test_default_counterpart_claim_updates_projection_identity(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log)
    projection = CounterpartProjection(store)
    router = CounterpartRouter(log, counterpart_projection=projection)

    local = router.upsert_from_source_metadata(
        {"channel": "cli", "command": "ask"},
        emitter=emitter,
    )
    first_channel_user = router.upsert_from_source_metadata(
        {
            "channel": "gateway",
            "platform": "slack",
            "user_id": "u1",
            "user_name": "Eric",
        },
        emitter=emitter,
    )

    stored = projection.get(DEFAULT_COUNTERPART_ID)
    assert stored is not None
    assert local == first_channel_user
    assert stored.identity["platform"] == "slack"
    assert stored.identity["user_id"] == "u1"
    assert stored.identity["display_name"] == "Eric"


def test_counterpart_router_distinguishes_channel_users_after_default_claim() -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log)
    router = CounterpartRouter(log)

    first = router.upsert_from_source_metadata(
        {"channel": "gateway", "platform": "slack", "user_id": "u1"},
        emitter=emitter,
    )
    same_first = router.upsert_from_source_metadata(
        {"channel": "gateway", "platform": "slack", "user_id": "u1"},
        emitter=emitter,
    )
    second = router.upsert_from_source_metadata(
        {"channel": "gateway", "platform": "slack", "user_id": "u2"},
        emitter=emitter,
    )

    assert first is not None
    assert first.id == str(DEFAULT_COUNTERPART_ID)
    assert same_first == first
    assert second is not None
    assert second != first


def test_counterpart_router_first_observed_no_duplicate_and_perception_source() -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log)
    router = CounterpartRouter(log)
    source_metadata = {"platform": "test", "user_id": "u1"}

    first = router.upsert_from_source_metadata(source_metadata, emitter=emitter)
    second = router.upsert_from_source_metadata(source_metadata, emitter=emitter)

    assert first is not None
    assert first == second
    assert [
        event.kind for event in log.iter(kinds=[CognitiveEventKind.COUNTERPART_FIRST_OBSERVED])
    ] == [CognitiveEventKind.COUNTERPART_FIRST_OBSERVED]

    thread_id = ThreadId.from_session("s1", source_metadata)
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=_StaticProvider(),
        tools=build_tool_registry(),
        emitter=emitter,
    )
    controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=first,
            payload="from user",
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=thread_id,
    )

    perceived = [event for event in log.iter(kinds=[CognitiveEventKind.PERCEIVED])][0]
    assert perceived.payload["from_counterpart"] == first.to_record()


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
