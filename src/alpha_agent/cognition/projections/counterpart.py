"""Counterpart projection backed by the shared SQLite database."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    Counterpart,
    CounterpartId,
    CounterpartRole,
    Relationship,
)
from alpha_agent.cognition.projections.base import EventProjection
from alpha_agent.state.store import StateStore


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _loads_list(value: str | None) -> list[Any]:
    if not value:
        return []
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else []


@dataclass(frozen=True)
class CounterpartProjectionView:
    counterparts: tuple[Counterpart, ...]


class CounterpartProjection(EventProjection):
    """Materialize counterpart lifecycle events into counterpart_view."""

    name = "counterpart"
    handles = frozenset(
        {
            CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
            CognitiveEventKind.COUNTERPART_IDENTIFIED,
            CognitiveEventKind.COUNTERPART_RELATIONSHIP_CHANGED,
            CognitiveEventKind.SERVICE_COMMITTED,
            CognitiveEventKind.SERVICE_FULFILLED,
            CognitiveEventKind.SERVICE_FAILED,
            CognitiveEventKind.TRUST_UPDATED,
        }
    )

    def __init__(self, store: StateStore):
        self.store = store

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind not in self.handles:
            return
        counterpart_id = self._counterpart_id(event)
        if event.kind == CognitiveEventKind.COUNTERPART_FIRST_OBSERVED:
            self._first_observed(event, counterpart_id)
        elif event.kind == CognitiveEventKind.COUNTERPART_IDENTIFIED:
            self._merge_fields(event, counterpart_id, identity=event.payload.get("identity"))
        elif event.kind == CognitiveEventKind.COUNTERPART_RELATIONSHIP_CHANGED:
            self._merge_fields(
                event,
                counterpart_id,
                relationship=event.payload.get("relationship", "observed"),
            )
        elif event.kind == CognitiveEventKind.TRUST_UPDATED:
            self._merge_fields(event, counterpart_id, trust_level=event.payload.get("trust_level"))
        elif event.kind in {
            CognitiveEventKind.SERVICE_COMMITTED,
            CognitiveEventKind.SERVICE_FULFILLED,
            CognitiveEventKind.SERVICE_FAILED,
        }:
            self._append_service_event(event, counterpart_id)

    def reset(self) -> None:
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM counterpart_view")

    def view(self) -> CounterpartProjectionView:
        return CounterpartProjectionView(counterparts=tuple(self.list_active()))

    def get(self, counterpart_id: CounterpartId | str) -> Counterpart | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM counterpart_view WHERE id = ?",
                (str(counterpart_id),),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_active(self) -> list[Counterpart]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM counterpart_view
                ORDER BY last_interaction_at DESC, id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def by_role(self, role: CounterpartRole) -> list[Counterpart]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM counterpart_view
                WHERE role = ?
                ORDER BY last_interaction_at DESC, id ASC
                """,
                (role.value,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _first_observed(self, event: CognitiveEvent, counterpart_id: str) -> None:
        role = CounterpartRole(event.payload.get("role", CounterpartRole.ANONYMOUS.value))
        identity = event.payload.get("identity") or {}
        metadata = event.payload.get("metadata") or {}
        relationship = event.payload.get("relationship", "observed")
        trust_level = float(event.payload.get("trust_level", 0.5))
        communication_style = event.payload.get("communication_style") or []
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO counterpart_view
                    (id, role, identity, relationship, service_contract, trust_level,
                     communication_style, first_seen_at, last_interaction_at, metadata,
                     last_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role = excluded.role,
                    identity = excluded.identity,
                    relationship = excluded.relationship,
                    trust_level = excluded.trust_level,
                    communication_style = excluded.communication_style,
                    last_interaction_at = excluded.last_interaction_at,
                    metadata = excluded.metadata,
                    last_event_id = excluded.last_event_id
                """,
                (
                    counterpart_id,
                    role.value,
                    _dumps(identity),
                    relationship,
                    "[]",
                    trust_level,
                    _dumps(communication_style),
                    str(event.timestamp),
                    str(event.timestamp),
                    _dumps(metadata),
                    str(event.id),
                ),
            )

    def _merge_fields(
        self,
        event: CognitiveEvent,
        counterpart_id: str,
        *,
        identity: Any | None = None,
        relationship: Any | None = None,
        trust_level: Any | None = None,
    ) -> None:
        existing = self.get(counterpart_id)
        if existing is None:
            self._first_observed(event, counterpart_id)
            existing = self.get(counterpart_id)
        if existing is None:
            raise RuntimeError(f"failed to materialize counterpart: {counterpart_id}")
        new_identity = dict(existing.identity)
        if isinstance(identity, dict):
            new_identity.update(identity)
        new_relationship = (
            str(relationship) if relationship is not None else existing.relationship.kind
        )
        new_trust = float(trust_level) if trust_level is not None else existing.trust_level
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE counterpart_view
                SET identity = ?,
                    relationship = ?,
                    trust_level = ?,
                    last_interaction_at = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (
                    _dumps(new_identity),
                    new_relationship,
                    new_trust,
                    str(event.timestamp),
                    str(event.id),
                    counterpart_id,
                ),
            )

    def _append_service_event(self, event: CognitiveEvent, counterpart_id: str) -> None:
        existing = self.get(counterpart_id)
        if existing is None:
            self._first_observed(event, counterpart_id)
            existing = self.get(counterpart_id)
        service_contract = (
            [item.to_record() for item in existing.service_contract] if existing is not None else []
        )
        service_payload = dict(event.payload.get("service") or {})
        service_payload.setdefault("id", f"{event.kind.value}:{event.id}")
        service_payload.setdefault(
            "description",
            event.payload.get("description", event.kind.value),
        )
        service_payload.setdefault("status", event.kind.value)
        service_payload.setdefault("metadata", {})
        service_payload["event_kind"] = event.kind.value
        service_payload["event_id"] = str(event.id)
        service_contract.append(service_payload)
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE counterpart_view
                SET service_contract = ?,
                    last_interaction_at = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (_dumps(service_contract), str(event.timestamp), str(event.id), counterpart_id),
            )

    def _counterpart_id(self, event: CognitiveEvent) -> str:
        value = event.payload.get("counterpart_id") or event.payload.get("id")
        if value is None:
            for ref in [*event.inputs, *event.outputs]:
                if ref.kind == "counterpart":
                    return ref.id
        if value is None:
            raise ValueError(f"counterpart event missing counterpart_id: {event.id}")
        return str(value)

    def _from_row(self, row: Any) -> Counterpart:
        record = {
            "id": row["id"],
            "role": row["role"],
            "identity": _loads_dict(row["identity"]),
            "relationship": {"kind": row["relationship"], "notes": "", "metadata": {}},
            "service_contract": _loads_list(row["service_contract"]),
            "trust_level": float(row["trust_level"]),
            "communication_style": _loads_list(row["communication_style"]),
            "first_seen_at": row["first_seen_at"],
            "last_interaction_at": row["last_interaction_at"],
            "metadata": _loads_dict(row["metadata"]),
        }
        counterpart = Counterpart.from_record(record)
        if isinstance(counterpart.relationship, str):
            return Counterpart(
                id=counterpart.id,
                role=counterpart.role,
                identity=counterpart.identity,
                relationship=Relationship(kind=counterpart.relationship),
                service_contract=counterpart.service_contract,
                trust_level=counterpart.trust_level,
                communication_style=counterpart.communication_style,
                first_seen_at=counterpart.first_seen_at,
                last_interaction_at=counterpart.last_interaction_at,
                metadata=counterpart.metadata,
            )
        return counterpart
