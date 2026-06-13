from __future__ import annotations

from alpha_agent.state.store import StateStore


def test_list_session_records_orders_by_durable_created_at_then_session_id(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.create_session_record(
        "session_late",
        created_at="2026-06-12T03:00:00+00:00",
    )
    store.create_session_record(
        "session_b",
        created_at="2026-06-12T01:00:00+00:00",
    )
    store.create_session_record(
        "session_a",
        created_at="2026-06-12T01:00:00+00:00",
    )

    records = store.list_session_records()

    assert [(record.session_id, record.created_at) for record in records] == [
        ("session_a", "2026-06-12T01:00:00+00:00"),
        ("session_b", "2026-06-12T01:00:00+00:00"),
        ("session_late", "2026-06-12T03:00:00+00:00"),
    ]
