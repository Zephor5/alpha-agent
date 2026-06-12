from __future__ import annotations

import pytest

from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefScope,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    Role,
    SituationId,
    ValidityWindow,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.source_time import (
    render_source_time_line,
    resolve_belief_source_time_range,
    resolve_source_time_range,
)
from alpha_agent.state.store import StateStore


def test_source_time_range_uses_session_messages_for_utc_metadata_and_local_render(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="Asia/Shanghai")
    first = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="first",
        created_at="2026-06-12T01:00:00+00:00",
    )
    second = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="second",
        created_at="2026-06-12T01:17:00+00:00",
    )

    source_time = resolve_source_time_range(
        store,
        [Reference("session_message", second.id), Reference("session_message", first.id)],
    )

    assert source_time is not None
    assert source_time.to_metadata() == {
        "source_time_start": "2026-06-12T01:00:00+00:00",
        "source_time_end": "2026-06-12T01:17:00+00:00",
        "source_time_basis": "session_message",
    }
    assert (
        render_source_time_line(store, source_time)
        == "Source message time range: 2026-06-12 09:00 to 2026-06-12 09:17 "
        "(Asia/Shanghai)."
    )


def test_source_time_single_message_prompt_line(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="Asia/Shanghai")
    message = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="single",
        created_at="2026-06-12T01:00:00+00:00",
    )

    source_time = resolve_source_time_range(store, [Reference("session_message", message.id)])

    assert source_time is not None
    assert (
        render_source_time_line(store, source_time)
        == "Source message time: 2026-06-12 09:00 (Asia/Shanghai)."
    )


def test_source_time_skips_reminder_and_compressed_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="Asia/Shanghai")
    evidence = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="evidence",
        created_at="2026-06-12T01:00:00+00:00",
    )
    reminder = store.append_session_message(
        session_id="s1",
        kind="system_reminder",
        llm_role="user",
        raw_content="context only",
        created_at="2026-06-12T01:10:00+00:00",
    )
    compressed = store.append_compressed_message(
        session_id="s1",
        raw_content="handover",
        compression_point_ordinal=1,
        compression_version="v1",
        created_at="2026-06-12T01:17:00+00:00",
    )

    source_time = resolve_source_time_range(
        store,
        [
            Reference("session_message", evidence.id),
            Reference("session_message", reminder.id),
            Reference("session_message", compressed.id),
        ],
    )

    assert source_time is not None
    assert source_time.to_metadata() == {
        "source_time_start": "2026-06-12T01:00:00+00:00",
        "source_time_end": "2026-06-12T01:00:00+00:00",
        "source_time_basis": "session_message",
    }
    assert (
        render_source_time_line(store, source_time)
        == "Source message time: 2026-06-12 09:00 (Asia/Shanghai)."
    )


def test_source_time_ignores_non_session_refs(tmp_path) -> None:
    store = _store(tmp_path)

    assert resolve_source_time_range(store, [Reference("runtime_trace", "missing")]) is None


def test_source_time_missing_session_message_ref_fails_fast(tmp_path) -> None:
    store = _store(tmp_path)

    with pytest.raises(KeyError, match="missing session_message source refs: missing-message"):
        resolve_source_time_range(store, [Reference("session_message", "missing-message")])


def test_source_time_invalid_timestamp_fails_fast(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="Asia/Shanghai")
    message = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="bad timestamp",
        created_at="2026-06-12T01:00:00+00:00",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE session_messages SET created_at = ? WHERE id = ?",
            ("not-a-timestamp", message.id),
        )
        conn.commit()

    with pytest.raises(ValueError, match=f"invalid created_at for session_message {message.id}"):
        resolve_source_time_range(store, [Reference("session_message", message.id)])


def test_source_time_rendering_uses_stored_session_timezone(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="America/New_York")
    message = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="timezone",
        created_at="2026-06-12T01:00:00+00:00",
    )

    source_time = resolve_source_time_range(store, [Reference("session_message", message.id)])

    assert source_time is not None
    assert (
        render_source_time_line(store, source_time)
        == "Source message time: 2026-06-11 21:00 (America/New_York)."
    )


def test_belief_source_time_resolution_uses_session_message_sources_only(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_session_record("s1", timezone="Asia/Shanghai")
    message = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="belief evidence",
        created_at="2026-06-12T01:00:00+00:00",
    )
    belief = _belief(
        sources=[
            Reference("runtime_trace", "missing-runtime-trace"),
            Reference("session_message", message.id),
            Reference("belief", "belief:other"),
        ]
    )

    source_time = resolve_belief_source_time_range(store, belief)

    assert source_time is not None
    assert source_time.to_metadata() == {
        "source_time_start": "2026-06-12T01:00:00+00:00",
        "source_time_end": "2026-06-12T01:00:00+00:00",
        "source_time_basis": "session_message",
    }


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _belief(*, sources: list[Reference]) -> AtomicBelief:
    return AtomicBelief(
        id=BeliefId("belief:source-time"),
        subject=subject_ref(SUBJECT_SELF),
        about=[],
        object="source time",
        content=NLStatement("Source time is derived from session messages."),
        memory_kind=MemoryKind.FACT,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope.GLOBAL,
        authority=Authority.USER_ASSERTED,
        sources=sources,
        validity=ValidityWindow(observed_at=Instant("2026-06-12T00:00:00+00:00")),
        formed_in=situation_ref(SituationId("situation:source-time")),
        holder_role=Role("agent"),
        held_since=Instant("2026-06-12T00:00:00+00:00"),
    )
