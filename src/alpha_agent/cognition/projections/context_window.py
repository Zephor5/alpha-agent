"""Foreground context-window projection.

Raw perception payloads stay in the cognitive event log. The materialized
foreground stores only perception ids and other lightweight references; get()
reconstructs Perception objects from their original perceived events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    Belief,
    BeliefId,
    BeliefRef,
    CognitiveEvent,
    CognitiveEventKind,
    CompressedSummary,
    ContextWindow,
    CounterpartId,
    CounterpartRef,
    Instant,
    JudgmentRef,
    NLStatement,
    Perception,
    PerceptionId,
    ProcedureRef,
    SituationId,
    Subject,
    ThreadId,
    ThreadKind,
    belief_ref,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.utils.time import utc_now_iso


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_list(value: str) -> list[str]:
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


@dataclass
class _ThreadState:
    thread_id: ThreadId
    counterpart_id: str | None = None
    foreground_ids: list[str] = field(default_factory=list)
    anchored_ids: list[str] = field(default_factory=list)
    recent_judgment_ids: list[str] = field(default_factory=list)
    matched_procedure_ids: list[str] = field(default_factory=list)
    background_summary_id: str | None = None
    last_event_id: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ContextWindowProjectionView:
    thread_count: int
    status: str = "active"


class ContextWindowProjection(Projection):
    """Maintain foreground perception references per conversation/cognition thread."""

    name = "context_window"
    handles = frozenset(
        {
            CognitiveEventKind.PERCEIVED,
            CognitiveEventKind.JUDGED,
            CognitiveEventKind.PROCEDURE_MATCHED,
            CognitiveEventKind.CONTEXT_COMPRESSED,
            CognitiveEventKind.CONTEXT_ANCHOR_SET,
            CognitiveEventKind.CONTEXT_ANCHOR_CLEARED,
        }
    )
    status = "active"

    def __init__(
        self,
        event_log: EventLog,
        *,
        recent_limit: int = 12,
        recent_judgment_limit: int = 8,
    ):
        self.event_log = event_log
        self.recent_limit = max(1, recent_limit)
        self.recent_judgment_limit = max(1, recent_judgment_limit)
        self.anchor_limit = max(1, self.recent_limit // 2)
        self._memory_states: dict[str, _ThreadState] = {}
        self._store = event_log.store if isinstance(event_log, SQLiteEventLog) else None
        if self._store is not None:
            self._ensure_schema()
        self._replay_existing_events()

    def get(
        self,
        thread_id: ThreadId,
        subject: Subject,
        at: Instant | None = None,
    ) -> ContextWindow:
        state = self._state_at(thread_id, at) if at is not None else self._load_state(thread_id)
        foreground = self._foreground(thread_id, state, at)
        counterpart = self._counterpart_from_state(state)
        situation = (
            foreground[-1].situation
            if foreground
            else situation_ref(SituationId("situation:context-window-empty"))
        )
        return ContextWindow(
            thread_id=thread_id,
            counterpart=counterpart,
            foreground=foreground,
            background=CompressedSummary(state.background_summary_id)
            if state and state.background_summary_id
            else None,
            recalled=[],
            recent_judgments=_judgment_refs(state.recent_judgment_ids if state else []),
            matched_procedures=_procedure_refs(state.matched_procedure_ids if state else []),
            subject_at=subject_ref(subject.id),
            situation_at=situation,
            assembled_at=at or Instant(utc_now_iso()),
            metadata={
                "status": self.status,
                "foreground_ids": list(state.foreground_ids) if state else [],
                "anchored_ids": list(state.anchored_ids) if state else [],
            },
        )

    def list_threads_by_counterpart(self, counterpart: CounterpartRef) -> list[ThreadId]:
        if self._store is None:
            states = sorted(
                self._memory_states.values(),
                key=lambda state: (_thread_key(state.thread_id), state.updated_at),
            )
            return [
                state.thread_id
                for state in states
                if state.counterpart_id == counterpart.id
                and state.thread_id.kind == ThreadKind.CONVERSATION
            ]
        with self._store.connect() as conn:
            rows = conn.execute(
                """
                SELECT thread_id
                FROM context_window_view
                WHERE counterpart_id = ? AND thread_kind = ?
                ORDER BY updated_at ASC, thread_id ASC
                """,
                (counterpart.id, ThreadKind.CONVERSATION.value),
            ).fetchall()
        return [ThreadId.from_record(json.loads(row["thread_id"])) for row in rows]

    def attach_recalled(
        self,
        window: ContextWindow,
        recalled: list[Belief | BeliefRef],
    ) -> ContextWindow:
        return replace(window, recalled=[_belief_reference(item) for item in recalled])

    def mark_anchor(self, thread_id: ThreadId, perception_id: PerceptionId | str) -> None:
        state = self._load_state(thread_id) or _ThreadState(thread_id=thread_id)
        raw_id = str(perception_id)
        if raw_id not in state.anchored_ids and len(state.anchored_ids) >= self.anchor_limit:
            raise ValueError(
                f"context anchors for {thread_id.key!r} are limited to {self.anchor_limit}"
            )
        event = EventEmitter(self.event_log).emit(
            CognitiveEventKind.CONTEXT_ANCHOR_SET,
            rationale=NLStatement("Marked perception as foreground context anchor."),
            payload={"thread_id": thread_id.to_record(), "perception_id": raw_id},
        )
        self.apply(event)

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind not in self.handles:
            return
        thread_id = self._thread_from_event(event)
        if thread_id is None:
            return
        state = self._load_state(thread_id) or _ThreadState(thread_id=thread_id)
        self._apply_to_state(state, event)
        self._save_state(state)

    def reset(self) -> None:
        self._memory_states.clear()
        if self._store is not None:
            self._ensure_schema()
            with self._store.immediate_transaction() as conn:
                conn.execute("DELETE FROM context_window_view")

    def view(self) -> ContextWindowProjectionView:
        if self._store is None:
            return ContextWindowProjectionView(thread_count=len(self._memory_states))
        with self._store.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM context_window_view").fetchone()
        return ContextWindowProjectionView(thread_count=int(row["count"]))

    def _replay_existing_events(self) -> None:
        for event in self.event_log.iter(kinds=self.handles):
            self.apply(event)

    def _apply_to_state(self, state: _ThreadState, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.PERCEIVED:
            self._apply_perceived(state, event)
        elif event.kind == CognitiveEventKind.JUDGED:
            state.recent_judgment_ids = _append_limited(
                state.recent_judgment_ids,
                _reference_ids(event, "judgment"),
                self.recent_judgment_limit,
            )
        elif event.kind == CognitiveEventKind.PROCEDURE_MATCHED:
            state.matched_procedure_ids = _append_limited(
                state.matched_procedure_ids,
                _reference_ids(event, "procedure"),
                self.recent_judgment_limit,
            )
        elif event.kind == CognitiveEventKind.CONTEXT_COMPRESSED:
            summary_id = event.payload.get("background_summary_id")
            state.background_summary_id = str(summary_id) if summary_id else None
        elif event.kind == CognitiveEventKind.CONTEXT_ANCHOR_SET:
            perception_id = event.payload.get("perception_id")
            if perception_id is not None:
                _append_unique(state.anchored_ids, str(perception_id))
                _append_unique(state.foreground_ids, str(perception_id))
                state.foreground_ids = self._roll_foreground(state)
        elif event.kind == CognitiveEventKind.CONTEXT_ANCHOR_CLEARED:
            perception_id = event.payload.get("perception_id")
            if perception_id is not None:
                state.anchored_ids = [
                    item for item in state.anchored_ids if item != str(perception_id)
                ]
                state.foreground_ids = self._roll_foreground(state)
        state.last_event_id = str(event.id)
        state.updated_at = str(event.timestamp)

    def _apply_perceived(self, state: _ThreadState, event: CognitiveEvent) -> None:
        perception = self._perception_from_event(event)
        _append_unique(state.foreground_ids, str(perception.id))
        if state.thread_id.kind == ThreadKind.COGNITION:
            state.counterpart_id = None
        elif perception.from_counterpart is not None:
            state.counterpart_id = perception.from_counterpart.id
        state.foreground_ids = self._roll_foreground(state)

    def _roll_foreground(self, state: _ThreadState) -> list[str]:
        ordered = _dedupe(state.foreground_ids)
        anchored = [item for item in ordered if item in set(state.anchored_ids)]
        unanchored = [item for item in ordered if item not in set(state.anchored_ids)]
        slots = max(0, self.recent_limit - len(anchored))
        selected = set(anchored)
        selected.update(unanchored[-slots:] if slots else [])
        return [item for item in ordered if item in selected]

    def _foreground(
        self,
        thread_id: ThreadId,
        state: _ThreadState | None,
        at: Instant | None,
    ) -> list[Perception]:
        if state is None:
            return []
        by_id = self._perceptions_by_id(thread_id, at)
        return [by_id[item] for item in state.foreground_ids if item in by_id]

    def _perceptions_by_id(
        self,
        thread_id: ThreadId,
        at: Instant | None = None,
    ) -> dict[str, Perception]:
        result: dict[str, Perception] = {}
        for event in self.event_log.iter(kinds=[CognitiveEventKind.PERCEIVED], until=at):
            if self._thread_from_event(event) != thread_id:
                continue
            perception = self._perception_from_event(event)
            result[str(perception.id)] = perception
        return result

    def _state_at(self, thread_id: ThreadId, at: Instant) -> _ThreadState | None:
        state: _ThreadState | None = None
        for event in self.event_log.iter(kinds=self.handles, until=at):
            if self._thread_from_event(event) != thread_id:
                continue
            if state is None:
                state = _ThreadState(thread_id=thread_id)
            self._apply_to_state(state, event)
        return state

    def _load_state(self, thread_id: ThreadId) -> _ThreadState | None:
        key = _thread_key(thread_id)
        if self._store is None:
            return self._memory_states.get(key)
        with self._store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_window_view WHERE thread_id = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return _state_from_row(row)

    def _save_state(self, state: _ThreadState) -> None:
        key = _thread_key(state.thread_id)
        if self._store is None:
            self._memory_states[key] = state
            return
        with self._store.immediate_transaction() as conn:
            conn.execute(
                """
                INSERT INTO context_window_view
                    (thread_id, thread_kind, counterpart_id, foreground_ids, anchored_ids,
                     recent_judgment_ids, matched_procedure_ids, background_summary_id,
                     last_event_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    thread_kind = excluded.thread_kind,
                    counterpart_id = excluded.counterpart_id,
                    foreground_ids = excluded.foreground_ids,
                    anchored_ids = excluded.anchored_ids,
                    recent_judgment_ids = excluded.recent_judgment_ids,
                    matched_procedure_ids = excluded.matched_procedure_ids,
                    background_summary_id = excluded.background_summary_id,
                    last_event_id = excluded.last_event_id,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    state.thread_id.kind.value,
                    state.counterpart_id,
                    _dumps(state.foreground_ids),
                    _dumps(state.anchored_ids),
                    _dumps(state.recent_judgment_ids),
                    _dumps(state.matched_procedure_ids),
                    state.background_summary_id,
                    state.last_event_id,
                    state.updated_at,
                ),
            )

    def _ensure_schema(self) -> None:
        if self._store is None:
            return
        with self._store.immediate_transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_window_view (
                    thread_id TEXT PRIMARY KEY,
                    thread_kind TEXT NOT NULL,
                    counterpart_id TEXT,
                    foreground_ids TEXT NOT NULL DEFAULT '[]',
                    anchored_ids TEXT NOT NULL DEFAULT '[]',
                    recent_judgment_ids TEXT NOT NULL DEFAULT '[]',
                    matched_procedure_ids TEXT NOT NULL DEFAULT '[]',
                    background_summary_id TEXT,
                    last_event_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ctx_window_counterpart
                    ON context_window_view(counterpart_id, thread_kind);
                CREATE INDEX IF NOT EXISTS idx_ctx_window_kind
                    ON context_window_view(thread_kind);
                """
            )

    def _counterpart_from_state(self, state: _ThreadState | None) -> CounterpartRef | None:
        if (
            state is None
            or state.thread_id.kind == ThreadKind.COGNITION
            or state.counterpart_id is None
        ):
            return None
        return counterpart_ref(CounterpartId(state.counterpart_id))

    def _thread_from_event(self, event: CognitiveEvent) -> ThreadId | None:
        raw = event.payload.get("thread_id")
        return ThreadId.from_record(raw) if isinstance(raw, dict) else None

    def _perception_from_event(self, event: CognitiveEvent) -> Perception:
        raw = event.payload.get("perception")
        if not isinstance(raw, dict):
            raise ValueError(f"perceived event missing perception payload: {event.id}")
        return Perception.from_record(raw)


def _thread_key(thread_id: ThreadId) -> str:
    return _dumps(thread_id.to_record())


def _state_from_row(row: Any) -> _ThreadState:
    return _ThreadState(
        thread_id=ThreadId.from_record(json.loads(row["thread_id"])),
        counterpart_id=row["counterpart_id"],
        foreground_ids=_loads_list(row["foreground_ids"]),
        anchored_ids=_loads_list(row["anchored_ids"]),
        recent_judgment_ids=_loads_list(row["recent_judgment_ids"]),
        matched_procedure_ids=_loads_list(row["matched_procedure_ids"]),
        background_summary_id=row["background_summary_id"],
        last_event_id=row["last_event_id"],
        updated_at=row["updated_at"],
    )


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        _append_unique(result, item)
    return result


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _append_limited(existing: list[str], new_items: list[str], limit: int) -> list[str]:
    merged = list(existing)
    for item in new_items:
        _append_unique(merged, item)
    return merged[-limit:]


def _reference_ids(event: CognitiveEvent, kind: str) -> list[str]:
    return [ref.id for ref in event.outputs if ref.kind == kind]


def _judgment_refs(ids: list[str]) -> list[JudgmentRef]:
    return [JudgmentRef("judgment", item) for item in ids]


def _procedure_refs(ids: list[str]) -> list[ProcedureRef]:
    return [ProcedureRef("procedure", item) for item in ids]


def _belief_reference(value: Belief | BeliefRef) -> BeliefRef:
    if isinstance(value, Belief):
        return belief_ref(BeliefId(str(value.id)))
    return value
