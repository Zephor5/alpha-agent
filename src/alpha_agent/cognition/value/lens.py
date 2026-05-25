"""SQLite-backed subject value lens helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, NLStatement, ValueKind
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.models.value import ValueLens
from alpha_agent.state.store import StateStore
from alpha_agent.utils.time import utc_now_iso

DEFAULT_PRIORITIES: tuple[ValueKind, ...] = (
    ValueKind.SAFETY,
    ValueKind.HONESTY,
    ValueKind.HELPFULNESS,
    ValueKind.AUTONOMY,
    ValueKind.EFFICIENCY,
    ValueKind.LEARNING,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_value_lens (
    subject_id TEXT PRIMARY KEY,
    priority TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT '{}',
    tradeoff_preferences TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
)
"""


def ensure_lens_schema(store: StateStore) -> None:
    with store.transaction() as conn:
        conn.execute(_SCHEMA)


def default_value_lens() -> ValueLens:
    priorities = list(DEFAULT_PRIORITIES)
    return ValueLens(
        priorities=priorities,
        weights=_priority_weights(priorities),
        sensitivity={value: 1.0 for value in priorities},
    )


def load_lens(store: StateStore, subject_id: str = SUBJECT_SELF) -> ValueLens:
    ensure_lens_schema(store)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT priority, sensitivity FROM subject_value_lens WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
    if row is None:
        return default_value_lens()
    priorities = _parse_priorities(row["priority"])
    sensitivity = _parse_value_map(row["sensitivity"])
    return ValueLens(
        priorities=priorities,
        weights=_priority_weights(priorities),
        sensitivity={**{value: 1.0 for value in priorities}, **sensitivity},
    )


def save_lens(
    store: StateStore,
    emitter: EventEmitter,
    lens: ValueLens,
    *,
    subject_id: str = SUBJECT_SELF,
    trigger: str,
    before: ValueLens | None = None,
) -> CognitiveEvent:
    before_lens = before or load_lens(store, subject_id)
    normalized = normalize_lens(lens)
    event = emitter.emit(
        CognitiveEventKind.VALUE_LENS_SHIFTED,
        rationale=NLStatement("Updated subject value lens."),
        payload={
            "subject_id": subject_id,
            "before": lens_to_record(before_lens),
            "after": lens_to_record(normalized),
            "trigger": trigger,
        },
    )
    upsert_lens_event(store, event)
    return event


def upsert_lens_event(store: StateStore, event: CognitiveEvent) -> None:
    raw_after = event.payload.get("after")
    if not isinstance(raw_after, dict):
        return
    subject_id = str(event.payload.get("subject_id") or event.subject.id or SUBJECT_SELF)
    lens = normalize_lens(ValueLens.from_record(raw_after))
    ensure_lens_schema(store)
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO subject_value_lens
                (subject_id, priority, sensitivity, tradeoff_preferences, updated_at, last_event_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_id) DO UPDATE SET
                priority = excluded.priority,
                sensitivity = excluded.sensitivity,
                tradeoff_preferences = excluded.tradeoff_preferences,
                updated_at = excluded.updated_at,
                last_event_id = excluded.last_event_id
            """,
            (
                subject_id,
                json.dumps([value.value for value in lens.priorities], sort_keys=True),
                json.dumps(
                    {value.value: amount for value, amount in lens.sensitivity.items()},
                    sort_keys=True,
                ),
                json.dumps([], sort_keys=True),
                str(event.timestamp) if event.timestamp else utc_now_iso(),
                str(event.id),
            ),
        )


def normalize_lens(lens: ValueLens) -> ValueLens:
    priorities = _unique_priorities(lens.priorities or DEFAULT_PRIORITIES)
    sensitivity = {value: float(lens.sensitivity.get(value, 1.0)) for value in priorities}
    return ValueLens(
        priorities=priorities,
        weights=_priority_weights(priorities),
        sensitivity=sensitivity,
    )


def lens_to_record(lens: ValueLens) -> dict[str, object]:
    return normalize_lens(lens).to_record()


def _priority_weights(priorities: Iterable[ValueKind]) -> dict[ValueKind, float]:
    ordered = list(priorities)
    count = len(ordered)
    return {
        value: float(count - index)
        for index, value in enumerate(ordered)
    }


def _parse_priorities(raw: str) -> list[ValueKind]:
    loaded = json.loads(raw or "[]")
    if not isinstance(loaded, list):
        return list(DEFAULT_PRIORITIES)
    return _unique_priorities(ValueKind(str(item)) for item in loaded)


def _parse_value_map(raw: str) -> dict[ValueKind, float]:
    loaded = json.loads(raw or "{}")
    if not isinstance(loaded, dict):
        return {}
    parsed: dict[ValueKind, float] = {}
    for key, value in loaded.items():
        parsed[ValueKind(str(key))] = float(value)
    return parsed


def _unique_priorities(values: Iterable[ValueKind]) -> list[ValueKind]:
    seen: set[ValueKind] = set()
    ordered: list[ValueKind] = []
    for value in values:
        kind = value if isinstance(value, ValueKind) else ValueKind(str(value))
        if kind in seen:
            continue
        seen.add(kind)
        ordered.append(kind)
    for value in DEFAULT_PRIORITIES:
        if value not in seen:
            ordered.append(value)
    return ordered
