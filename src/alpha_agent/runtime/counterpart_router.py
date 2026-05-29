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

DEFAULT_COUNTERPART_ID = CounterpartId("counterpart:main-user")
DEFAULT_COUNTERPART_IDENTITY = {
    "platform": "local",
    "user_id": "main_user",
    "display_name": "Main user",
}


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
        source_identity = self._source_identity(source_metadata)
        identity = source_identity or dict(DEFAULT_COUNTERPART_IDENTITY)
        counterpart_id = self._routed_counterpart_id(
            identity,
            is_default_source=source_identity is None,
        )
        ref = counterpart_ref(counterpart_id)
        if self._already_observed(counterpart_id):
            self._identify_default_counterpart(
                counterpart_id,
                ref,
                source_identity,
                emitter=emitter,
            )
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

    def _source_identity(self, source_metadata: Mapping[str, Any] | None) -> dict[str, str] | None:
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
        display_name = (
            source_metadata.get("display_name")
            or source_metadata.get("username")
            or source_metadata.get("user_name")
        )
        if display_name is not None:
            identity["display_name"] = str(display_name)
        return identity

    def _routed_counterpart_id(
        self,
        identity: Mapping[str, str],
        *,
        is_default_source: bool,
    ) -> CounterpartId:
        if is_default_source:
            return DEFAULT_COUNTERPART_ID
        if self._matches_default_counterpart(identity):
            return DEFAULT_COUNTERPART_ID
        if self._default_counterpart_can_be_claimed():
            return DEFAULT_COUNTERPART_ID
        return self._counterpart_id(identity["platform"], identity["user_id"])

    def _matches_default_counterpart(self, identity: Mapping[str, str]) -> bool:
        existing = self._counterpart_identity(DEFAULT_COUNTERPART_ID)
        return existing is not None and _same_source_identity(existing, identity)

    def _default_counterpart_can_be_claimed(self) -> bool:
        existing = self._counterpart_identity(DEFAULT_COUNTERPART_ID)
        if existing is None:
            return True
        return _same_source_identity(existing, DEFAULT_COUNTERPART_IDENTITY)

    def _identify_default_counterpart(
        self,
        counterpart_id: CounterpartId,
        ref: CounterpartRef,
        identity: Mapping[str, str] | None,
        *,
        emitter: EventEmitter,
    ) -> None:
        if counterpart_id != DEFAULT_COUNTERPART_ID or identity is None:
            return
        existing = self._counterpart_identity(DEFAULT_COUNTERPART_ID)
        if existing is not None and _same_source_identity(existing, identity):
            return
        event = emitter.emit(
            CognitiveEventKind.COUNTERPART_IDENTIFIED,
            outputs=[ref],
            payload={
                "counterpart_id": str(DEFAULT_COUNTERPART_ID),
                "identity": dict(identity),
            },
        )
        if self.counterpart_projection is not None:
            self.counterpart_projection.apply(event)

    def _counterpart_identity(self, counterpart_id: CounterpartId) -> dict[str, Any] | None:
        if self.counterpart_projection is not None:
            counterpart = self.counterpart_projection.get(counterpart_id)
            if counterpart is not None:
                return dict(counterpart.identity)
        identity: dict[str, Any] | None = None
        for event in self.event_log.iter(
            kinds=[
                CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
                CognitiveEventKind.COUNTERPART_IDENTIFIED,
            ]
        ):
            event_counterpart_id = event.payload.get("counterpart_id") or event.payload.get("id")
            if str(event_counterpart_id) != str(counterpart_id):
                continue
            payload_identity = event.payload.get("identity")
            if not isinstance(payload_identity, Mapping):
                continue
            if identity is None:
                identity = {}
            identity.update(payload_identity)
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


def _same_source_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return str(left.get("platform")) == str(right.get("platform")) and str(
        left.get("user_id")
    ) == str(right.get("user_id"))
