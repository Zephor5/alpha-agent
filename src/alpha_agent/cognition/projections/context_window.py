"""Foreground context-window projection.

Raw perception payloads stay in the cognitive event log. The materialized
foreground stores only perception ids and other lightweight references; get()
reconstructs Perception objects from their original perceived events.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    NLStatement,
    Perception,
    PerceptionId,
    ProcedureRef,
    Reference,
    SituationId,
    StimulusKind,
    Subject,
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
class _SessionState:
    session_id: str
    counterpart_id: str | None = None
    foreground_ids: list[str] = field(default_factory=list)
    anchored_ids: list[str] = field(default_factory=list)
    matched_procedure_ids: list[str] = field(default_factory=list)
    background_summary_id: str | None = None
    last_event_id: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ContextWindowProjectionView:
    session_count: int
    status: str = "active"


class ContextWindowProjection(Projection):
    """Maintain foreground perception references per runtime session."""

    name = "context_window"
    handles = frozenset(
        {
            CognitiveEventKind.PERCEIVED,
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
    ):
        self.event_log = event_log
        self.recent_limit = max(1, recent_limit)
        self.anchor_limit = max(1, self.recent_limit // 2)
        self._memory_states: dict[str, _SessionState] = {}
        self._store = event_log.store if isinstance(event_log, SQLiteEventLog) else None
        if self._store is not None:
            self._ensure_schema()
        self._replay_existing_events()

    def get(
        self,
        session_id: str,
        subject: Subject,
        at: Instant | None = None,
    ) -> ContextWindow:
        state = (
            self._state_at(session_id, at)
            if at is not None
            else self._load_state(session_id)
        )
        foreground = self._foreground(session_id, state, at)
        counterpart = self._counterpart_from_state(state)
        situation = (
            foreground[-1].situation
            if foreground
            else situation_ref(SituationId("situation:context-window-empty"))
        )
        return ContextWindow(
            session_id=session_id,
            counterpart=counterpart,
            foreground=foreground,
            background=CompressedSummary(state.background_summary_id)
            if state and state.background_summary_id
            else None,
            recalled=[],
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

    def list_sessions_by_counterpart(self, counterpart: CounterpartRef) -> list[str]:
        if self._store is None:
            states = sorted(
                self._memory_states.values(),
                key=lambda state: (state.session_id, state.updated_at),
            )
            return [
                state.session_id
                for state in states
                if state.counterpart_id == counterpart.id
            ]
        with self._store.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM context_window_view
                WHERE counterpart_id = ?
                ORDER BY updated_at ASC, session_id ASC
                """,
                (counterpart.id,),
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def list_session_ids(self) -> list[str]:
        """Return materialized context-window session ids in stable order."""

        if self._store is None:
            states = sorted(
                self._memory_states.values(),
                key=lambda state: (state.session_id, state.updated_at),
            )
            return [state.session_id for state in states]
        with self._store.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM context_window_view
                ORDER BY updated_at ASC, session_id ASC
                """
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def foreground_ids(self, session_id: str) -> list[str]:
        state = self._load_state(session_id)
        return list(state.foreground_ids) if state is not None else []

    def attach_recalled(
        self,
        window: ContextWindow,
        recalled: Sequence[Belief | BeliefRef],
    ) -> ContextWindow:
        return replace(window, recalled=[_belief_reference(item) for item in recalled])

    def mark_anchor(self, session_id: str, perception_id: PerceptionId | str) -> None:
        state = self._load_state(session_id) or _SessionState(session_id=session_id)
        raw_id = str(perception_id)
        if raw_id not in state.anchored_ids and len(state.anchored_ids) >= self.anchor_limit:
            raise ValueError(
                f"context anchors for session {session_id!r} are limited to {self.anchor_limit}"
            )
        event = EventEmitter(self.event_log).emit(
            CognitiveEventKind.CONTEXT_ANCHOR_SET,
            rationale=NLStatement("Marked perception as foreground context anchor."),
            payload={"session_id": session_id, "perception_id": raw_id},
        )
        self.apply(event)

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind not in self.handles:
            return
        session_id = self._session_id_from_event(event)
        if session_id is None:
            return
        state = self._load_state(session_id) or _SessionState(session_id=session_id)
        self._apply_to_state(state, event)
        self._save_state(state)

    def reset(self) -> None:
        self._memory_states.clear()
        if self._store is not None:
            self._ensure_schema()
            with self._store.immediate_transaction() as conn:
                conn.execute("DELETE FROM context_window_background")
                conn.execute("DELETE FROM context_window_view")

    def view(self) -> ContextWindowProjectionView:
        if self._store is None:
            return ContextWindowProjectionView(session_count=len(self._memory_states))
        with self._store.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM context_window_view").fetchone()
        return ContextWindowProjectionView(session_count=int(row["count"]))

    def _replay_existing_events(self) -> None:
        for event in self.event_log.iter(kinds=self.handles):
            self.apply(event)

    def _apply_to_state(self, state: _SessionState, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.PERCEIVED:
            self._apply_perceived(state, event)
        elif event.kind == CognitiveEventKind.PROCEDURE_MATCHED:
            state.matched_procedure_ids = _append_limited(
                state.matched_procedure_ids,
                _reference_ids(event, "procedure"),
                self.recent_limit,
            )
        elif event.kind == CognitiveEventKind.CONTEXT_COMPRESSED:
            summary_id = event.payload.get("produced_summary_id") or event.payload.get(
                "background_summary_id"
            )
            absorbed_ids = {
                str(item) for item in event.payload.get("absorbed_perception_ids", [])
            }
            anchored_ids = set(state.anchored_ids)
            if absorbed_ids:
                state.foreground_ids = [
                    item
                    for item in state.foreground_ids
                    if item not in absorbed_ids or item in anchored_ids
                ]
            state.background_summary_id = str(summary_id) if summary_id else None
            self._save_background_summary(state, event, str(summary_id) if summary_id else None)
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

    def _apply_perceived(self, state: _SessionState, event: CognitiveEvent) -> None:
        perception = self._perception_from_event(event)
        _append_unique(state.foreground_ids, str(perception.id))
        if perception.from_counterpart is not None:
            state.counterpart_id = perception.from_counterpart.id
        state.foreground_ids = self._roll_foreground(state)

    def _roll_foreground(self, state: _SessionState) -> list[str]:
        ordered = _dedupe(state.foreground_ids)
        anchored = [item for item in ordered if item in set(state.anchored_ids)]
        unanchored = [item for item in ordered if item not in set(state.anchored_ids)]
        slots = max(0, self.recent_limit - len(anchored))
        selected = set(anchored)
        selected.update(unanchored[-slots:] if slots else [])
        return [item for item in ordered if item in selected]

    def _foreground(
        self,
        session_id: str,
        state: _SessionState | None,
        at: Instant | None,
    ) -> list[Perception]:
        if state is None:
            return []
        by_id = self._perceptions_by_id(session_id, at)
        return [by_id[item] for item in state.foreground_ids if item in by_id]

    def _perceptions_by_id(
        self,
        session_id: str,
        at: Instant | None = None,
    ) -> dict[str, Perception]:
        result: dict[str, Perception] = {}
        for event in self.event_log.iter(kinds=[CognitiveEventKind.PERCEIVED], until=at):
            if self._session_id_from_event(event) != session_id:
                continue
            perception = self._perception_from_event(event)
            result[str(perception.id)] = perception
        return result

    def _state_at(self, session_id: str, at: Instant) -> _SessionState | None:
        state: _SessionState | None = None
        for event in self.event_log.iter(kinds=self.handles, until=at):
            if self._session_id_from_event(event) != session_id:
                continue
            if state is None:
                state = _SessionState(session_id=session_id)
            self._apply_to_state(state, event)
        return state

    def _load_state(self, session_id: str) -> _SessionState | None:
        if self._store is None:
            return self._memory_states.get(session_id)
        with self._store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM context_window_view WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return _state_from_row(row)

    def _save_state(self, state: _SessionState) -> None:
        if self._store is None:
            self._memory_states[state.session_id] = state
            return
        with self._store.immediate_transaction() as conn:
            conn.execute(
                """
                INSERT INTO context_window_view
                    (session_id, counterpart_id, foreground_ids, anchored_ids,
                     matched_procedure_ids, background_summary_id,
                     last_event_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    counterpart_id = excluded.counterpart_id,
                    foreground_ids = excluded.foreground_ids,
                    anchored_ids = excluded.anchored_ids,
                    matched_procedure_ids = excluded.matched_procedure_ids,
                    background_summary_id = excluded.background_summary_id,
                    last_event_id = excluded.last_event_id,
                    updated_at = excluded.updated_at
                """,
                (
                    state.session_id,
                    state.counterpart_id,
                    _dumps(state.foreground_ids),
                    _dumps(state.anchored_ids),
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
                DROP TABLE IF EXISTS context_window_view;
                CREATE TABLE context_window_view (
                    session_id TEXT PRIMARY KEY,
                    counterpart_id TEXT,
                    foreground_ids TEXT NOT NULL DEFAULT '[]',
                    anchored_ids TEXT NOT NULL DEFAULT '[]',
                    matched_procedure_ids TEXT NOT NULL DEFAULT '[]',
                    background_summary_id TEXT,
                    last_event_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ctx_window_counterpart
                    ON context_window_view(counterpart_id);
                DROP TABLE IF EXISTS context_window_background;
                CREATE TABLE context_window_background (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    derived_from_perception_ids TEXT NOT NULL DEFAULT '[]',
                    preserved_anchors TEXT NOT NULL DEFAULT '[]',
                    compression_policy TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ctx_bg_session_time
                    ON context_window_background(session_id, created_at DESC);
                """
            )

    def _save_background_summary(
        self,
        state: _SessionState,
        event: CognitiveEvent,
        summary_id: str | None,
    ) -> None:
        if self._store is None or summary_id is None or "summary" not in event.payload:
            return
        with self._store.immediate_transaction() as conn:
            conn.execute(
                """
                INSERT INTO context_window_background
                    (id, session_id, summary, derived_from_perception_ids, preserved_anchors,
                     compression_policy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    summary = excluded.summary,
                    derived_from_perception_ids = excluded.derived_from_perception_ids,
                    preserved_anchors = excluded.preserved_anchors,
                    compression_policy = excluded.compression_policy,
                    created_at = excluded.created_at
                """,
                (
                    summary_id,
                    state.session_id,
                    str(event.payload.get("summary", "")),
                    _dumps(
                        [str(item) for item in event.payload.get("absorbed_perception_ids", [])]
                    ),
                    _dumps([str(item) for item in event.payload.get("preserved_anchors", [])]),
                    str(event.payload.get("compression_policy", "deterministic_v1")),
                    str(event.timestamp),
                ),
            )

    def _counterpart_from_state(self, state: _SessionState | None) -> CounterpartRef | None:
        if state is None or state.counterpart_id is None:
            return None
        return counterpart_ref(CounterpartId(state.counterpart_id))

    def _session_id_from_event(self, event: CognitiveEvent) -> str | None:
        raw = event.payload.get("session_id")
        if isinstance(raw, str) and raw.strip():
            return raw
        return None

    def _perception_from_event(self, event: CognitiveEvent) -> Perception:
        raw = event.payload.get("perception")
        if isinstance(raw, dict):
            return Perception.from_record(raw)
        turn_id = str(event.payload.get("turn_id") or event.id)
        source_kind = _stimulus_kind(event.payload.get("stimulus_kind"))
        counterpart = _reference_from_record(event.payload.get("from_counterpart"))
        situation = event.situation or situation_ref(SituationId(f"situation:{turn_id}"))
        return Perception(
            id=PerceptionId(_perception_id_from_event(event, turn_id)),
            source_kind=source_kind,
            from_counterpart=counterpart,
            raw=self._raw_from_source_message(event),
            surface_intent=[],
            raised_entities=[counterpart] if counterpart is not None else [],
            subject=event.subject,
            situation=situation,
            received_at=event.timestamp,
        )

    def _raw_from_source_message(self, event: CognitiveEvent) -> str:
        if self._store is None:
            return ""
        message_ids = [
            str(item.get("id"))
            for item in event.payload.get("source_refs", [])
            if isinstance(item, dict) and item.get("kind") == "session_message" and item.get("id")
        ]
        if not message_ids:
            return ""
        messages = self._store.list_session_messages_by_ids(message_ids[:1])
        if not messages:
            return ""
        return messages[0].raw_content

def _state_from_row(row: Any) -> _SessionState:
    return _SessionState(
        session_id=row["session_id"],
        counterpart_id=row["counterpart_id"],
        foreground_ids=_loads_list(row["foreground_ids"]),
        anchored_ids=_loads_list(row["anchored_ids"]),
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


def _procedure_refs(ids: list[str]) -> list[ProcedureRef]:
    return [ProcedureRef("procedure", item) for item in ids]


def _belief_reference(value: Belief | BeliefRef) -> BeliefRef:
    if isinstance(value, Belief):
        return belief_ref(BeliefId(str(value.id)))
    return value


def _perception_id_from_event(event: CognitiveEvent, turn_id: str) -> str:
    for ref in event.outputs:
        if getattr(ref, "kind", None) == "perception" and getattr(ref, "id", None):
            return str(ref.id)
    return f"perception:{turn_id}"


def _reference_from_record(value: object) -> Reference | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    ref_id = value.get("id")
    if not isinstance(kind, str) or not isinstance(ref_id, str):
        return None
    return Reference(kind, ref_id)


def _stimulus_kind(value: object) -> StimulusKind:
    if isinstance(value, str):
        try:
            return StimulusKind(value)
        except ValueError:
            return StimulusKind.USER_MESSAGE
    return StimulusKind.USER_MESSAGE
