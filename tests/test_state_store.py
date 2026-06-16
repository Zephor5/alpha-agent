from __future__ import annotations

import pytest

from alpha_agent.state.models import LLMUsageRecord
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


def test_session_usage_counters_default_to_zero(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    record = store.create_session_record("s1", created_at="2026-06-12T01:00:00+00:00")

    assert record.total_tokens == 0
    assert record.cached_tokens == 0
    assert record.prompt_cache_miss_tokens == 0
    assert record.reasoning_tokens == 0
    assert record.completion_tokens == 0
    assert record.occupied_tokens == 0


def test_add_session_usage_increments_cumulative_counters_without_touching_updated_at(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.create_session_record(
        "s1",
        created_at="2026-06-12T01:00:00+00:00",
        updated_at="2026-06-12T02:00:00+00:00",
    )

    store.add_session_usage(
        "s1",
        LLMUsageRecord(
            total_tokens=10,
            cached_tokens=3,
            prompt_cache_miss_tokens=4,
            reasoning_tokens=1,
            completion_tokens=2,
        ),
    )
    record = store.add_session_usage(
        "s1",
        LLMUsageRecord(
            total_tokens=7,
            cached_tokens=1,
            prompt_cache_miss_tokens=2,
            reasoning_tokens=3,
            completion_tokens=4,
        ),
    )

    assert record.total_tokens == 17
    assert record.cached_tokens == 4
    assert record.prompt_cache_miss_tokens == 6
    assert record.reasoning_tokens == 4
    assert record.completion_tokens == 6
    assert record.occupied_tokens == 0
    assert record.updated_at == "2026-06-12T02:00:00+00:00"


def test_add_session_usage_raises_for_missing_session(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    with pytest.raises(KeyError):
        store.add_session_usage(
            "missing",
            LLMUsageRecord(
                total_tokens=1,
                cached_tokens=0,
                prompt_cache_miss_tokens=1,
                reasoning_tokens=0,
                completion_tokens=0,
            ),
        )


def test_update_session_occupied_tokens_can_decrease_without_mutating_cumulative_counters(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.create_session_record("s1")
    store.add_session_usage(
        "s1",
        LLMUsageRecord(
            total_tokens=100,
            cached_tokens=20,
            prompt_cache_miss_tokens=50,
            reasoning_tokens=10,
            completion_tokens=20,
        ),
    )

    store.update_session_occupied_tokens("s1", 80)
    record = store.update_session_occupied_tokens("s1", 12)

    assert record.occupied_tokens == 12
    assert record.total_tokens == 100
    assert record.cached_tokens == 20
    assert record.prompt_cache_miss_tokens == 50
    assert record.reasoning_tokens == 10
    assert record.completion_tokens == 20


def test_append_llm_call_preserves_id_raw_usage_and_allows_null_session_id(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    record = store.append_llm_call(
        id="llm_call_abc",
        session_id=None,
        worker_name="memory_extraction",
        provider="deepseek",
        model="deepseek-chat",
        usage=LLMUsageRecord(
            total_tokens=466,
            cached_tokens=256,
            prompt_cache_miss_tokens=106,
            reasoning_tokens=96,
            completion_tokens=104,
        ),
        raw_usage={
            "completion_tokens": 104,
            "nested": {"reasoning_tokens": 96},
            "ignored_full_response_field": None,
        },
        started_trace_id="trace_start",
        completed_trace_id="trace_done",
        created_at="2026-06-12T01:00:00+00:00",
    )

    assert record.id == "llm_call_abc"
    assert record.session_id is None
    assert record.worker_name == "memory_extraction"
    assert record.provider == "deepseek"
    assert record.model == "deepseek-chat"
    assert record.total_tokens == 466
    assert record.cached_tokens == 256
    assert record.prompt_cache_miss_tokens == 106
    assert record.reasoning_tokens == 96
    assert record.completion_tokens == 104
    assert record.raw_usage == {
        "completion_tokens": 104,
        "nested": {"reasoning_tokens": 96},
        "ignored_full_response_field": None,
    }
    assert record.started_trace_id == "trace_start"
    assert record.completed_trace_id == "trace_done"
    assert record.created_at == "2026-06-12T01:00:00+00:00"


def test_append_llm_call_uses_zero_token_defaults(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    record = store.append_llm_call(
        id="llm_call_zero",
        provider="mock",
        model="mock-model",
        created_at="2026-06-12T01:00:00+00:00",
    )

    assert record.total_tokens == 0
    assert record.cached_tokens == 0
    assert record.prompt_cache_miss_tokens == 0
    assert record.reasoning_tokens == 0
    assert record.completion_tokens == 0
    assert record.raw_usage == {}


def test_llm_calls_schema_matches_usage_ledger_contract(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    with store.connect() as conn:
        columns = conn.execute("PRAGMA table_info(llm_calls)").fetchall()
        column_names = [row["name"] for row in columns]
        indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list(llm_calls)").fetchall()
        }
        foreign_keys = conn.execute("PRAGMA foreign_key_list(llm_calls)").fetchall()
        conn.execute(
            """
            INSERT INTO llm_calls (id, provider, model, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("call_defaults", "mock", "mock", "2026-06-12T01:00:00+00:00"),
        )
        row = conn.execute(
            """
            SELECT *
            FROM llm_calls
            WHERE id = ?
            """,
            ("call_defaults",),
        ).fetchone()

    assert column_names == [
        "id",
        "session_id",
        "worker_name",
        "provider",
        "model",
        "total_tokens",
        "cached_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
        "completion_tokens",
        "raw_usage",
        "started_trace_id",
        "completed_trace_id",
        "created_at",
    ]
    assert {
        "idx_llm_calls_created_at",
        "idx_llm_calls_session_created",
        "idx_llm_calls_worker_created",
    }.issubset(indexes)
    assert foreign_keys == []
    assert row["session_id"] is None
    assert row["worker_name"] is None
    assert row["total_tokens"] == 0
    assert row["cached_tokens"] == 0
    assert row["prompt_cache_miss_tokens"] == 0
    assert row["reasoning_tokens"] == 0
    assert row["completion_tokens"] == 0
    assert row["raw_usage"] == "{}"


def test_list_llm_calls_filters_and_orders_by_created_at_then_id(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    usage = LLMUsageRecord(
        total_tokens=1,
        cached_tokens=0,
        prompt_cache_miss_tokens=1,
        reasoning_tokens=0,
        completion_tokens=0,
    )

    store.append_llm_call(
        id="call_b",
        session_id="s1",
        provider="mock",
        model="mock",
        usage=usage,
        created_at="2026-06-12T01:00:00+00:00",
    )
    store.append_llm_call(
        id="call_a",
        session_id="s1",
        provider="mock",
        model="mock",
        usage=usage,
        created_at="2026-06-12T01:00:00+00:00",
    )
    store.append_llm_call(
        id="call_worker",
        session_id=None,
        worker_name="memory_summary",
        provider="mock",
        model="mock",
        usage=usage,
        created_at="2026-06-12T02:00:00+00:00",
    )
    store.append_llm_call(
        id="call_other",
        session_id="s2",
        worker_name="memory_summary",
        provider="mock",
        model="mock",
        usage=usage,
        created_at="2026-06-12T03:00:00+00:00",
    )

    assert [call.id for call in store.list_llm_calls()] == [
        "call_a",
        "call_b",
        "call_worker",
        "call_other",
    ]
    assert [call.id for call in store.list_llm_calls(session_id="s1")] == [
        "call_a",
        "call_b",
    ]
    assert [call.id for call in store.list_llm_calls(worker_name="memory_summary")] == [
        "call_worker",
        "call_other",
    ]
    assert [call.id for call in store.list_llm_calls(session_id_is_null=True)] == [
        "call_worker",
    ]
    assert [call.id for call in store.list_llm_calls(limit=2)] == ["call_a", "call_b"]
