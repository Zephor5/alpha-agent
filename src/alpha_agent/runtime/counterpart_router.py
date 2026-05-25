"""Route source metadata to stable counterpart references."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartId,
    CounterpartRef,
    CounterpartRole,
    counterpart_ref,
)
from alpha_agent.cognition.projections.counterpart import CounterpartProjection


class CounterpartRouter:
    """Resolve platform source metadata into stable CounterpartRef values."""

    def __init__(
        self,
        event_log: EventLog,
        *,
        counterpart_projection: CounterpartProjection | None = None,
    ):
        self.event_log = event_log
        self.counterpart_projection = counterpart_projection

    def upsert_from_source_metadata(
        self,
        source_metadata: Mapping[str, Any] | None,
        *,
        emitter: EventEmitter,
    ) -> CounterpartRef | None:
        identity = self._identity(source_metadata)
        if identity is None:
            return None
        counterpart_id = self._counterpart_id(identity["platform"], identity["user_id"])
        ref = counterpart_ref(counterpart_id)
        if self._already_observed(counterpart_id):
            return ref

        role = self._role(source_metadata)
        event = emitter.emit(
            CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
            outputs=[ref],
            payload={
                "counterpart_id": str(counterpart_id),
                "role": role.value,
                "identity": identity,
                "relationship": "observed",
                "trust_level": 0.5,
                "metadata": dict(source_metadata or {}),
            },
        )
        if self.counterpart_projection is not None:
            self.counterpart_projection.apply(event)
        return ref

    def _already_observed(self, counterpart_id: CounterpartId) -> bool:
        if self.counterpart_projection is not None:
            if self.counterpart_projection.get(counterpart_id) is not None:
                return True
        for event in self.event_log.iter(kinds=[CognitiveEventKind.COUNTERPART_FIRST_OBSERVED]):
            if str(event.payload.get("counterpart_id")) == str(counterpart_id):
                return True
            for ref in [*event.inputs, *event.outputs]:
                if ref.kind == "counterpart" and ref.id == str(counterpart_id):
                    return True
        return False

    def _identity(self, source_metadata: Mapping[str, Any] | None) -> dict[str, str] | None:
        if not source_metadata:
            return None
        platform = source_metadata.get("platform") or source_metadata.get("source") or "local"
        user_id = (
            source_metadata.get("user_id")
            or source_metadata.get("sender_id")
            or source_metadata.get("account_id")
            or source_metadata.get("user")
        )
        if user_id is None:
            return None
        identity = {"platform": str(platform), "user_id": str(user_id)}
        display_name = source_metadata.get("display_name") or source_metadata.get("username")
        if display_name is not None:
            identity["display_name"] = str(display_name)
        return identity

    def _counterpart_id(self, platform: str, user_id: str) -> CounterpartId:
        digest = hashlib.sha1(f"{platform}:{user_id}".encode()).hexdigest()[:16]
        return CounterpartId(f"counterpart:{digest}")

    def _role(self, source_metadata: Mapping[str, Any] | None) -> CounterpartRole:
        role = str((source_metadata or {}).get("role") or CounterpartRole.USER.value)
        try:
            return CounterpartRole(role)
        except ValueError:
            return CounterpartRole.USER
