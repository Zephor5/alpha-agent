from __future__ import annotations

import pytest

from alpha_agent.state.store import (
    REMINDER_TYPE_COUNTERPART_PROFILE,
    REMINDER_TYPE_SELF_MEMORY_SUMMARY,
    REMINDER_TYPE_SESSION_TIME,
    StateStore,
)
from alpha_agent.utils.system_reminder import inline_system_reminder


def test_typed_reminder_lookup_keeps_time_and_stable_context_separate(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    first_time = store.append_session_time_reminder(
        session_id="s1",
        raw_content=inline_system_reminder("time update: 2026-06-12T09:00+08:00"),
        reminder_kind="time_update",
        local_datetime="2026-06-12T09:00+08:00",
        local_date="2026-06-12",
    )
    profile = store.append_session_reminder(
        session_id="s1",
        raw_content="Counterpart profile: User prefers concise answers.",
        reminder_type=REMINDER_TYPE_COUNTERPART_PROFILE,
    )
    self_memory = store.append_session_reminder(
        session_id="s1",
        raw_content="Self memory summary: Agent checks root causes.",
        reminder_type=REMINDER_TYPE_SELF_MEMORY_SUMMARY,
    )

    assert store.find_latest_session_time_reminder("s1") == first_time
    assert first_time.metadata["reminder_type"] == REMINDER_TYPE_SESSION_TIME
    assert (
        store.find_latest_session_reminder(
            "s1",
            reminder_type=REMINDER_TYPE_COUNTERPART_PROFILE,
        )
        == profile
    )
    assert store.list_session_reminders(
        "s1",
        reminder_type=REMINDER_TYPE_SELF_MEMORY_SUMMARY,
    ) == [self_memory]


def test_reminder_lookup_requires_known_concrete_type(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    with pytest.raises(ValueError, match="unsupported reminder_type"):
        store.find_latest_session_reminder("s1", reminder_type="system_reminder")

    with pytest.raises(ValueError, match="unsupported reminder_type"):
        store.list_session_reminders("s1", reminder_type="")
