"""Thread routing policies for cognition stimuli."""

from __future__ import annotations

from typing import Any

from alpha_agent.cognition.models import SUBJECT_SELF, Stimulus, StimulusKind, SubjectId, ThreadId


class StimulusRouter:
    """Map a stimulus to the foreground context thread it should update."""

    _CONVERSATION_KINDS = frozenset(
        {
            StimulusKind.USER_MESSAGE,
            StimulusKind.TOOL_RESULT,
            StimulusKind.WEBHOOK,
            StimulusKind.INTER_AGENT,
        }
    )

    @staticmethod
    def route(
        stimulus: Stimulus,
        *,
        session_id: str | None = None,
        subject_id: SubjectId = SUBJECT_SELF,
    ) -> ThreadId:
        return StimulusRouter.route_kind(
            stimulus.kind,
            payload=stimulus.payload,
            session_id=session_id,
            subject_id=subject_id,
        )

    @staticmethod
    def route_kind(
        kind: StimulusKind,
        *,
        payload: Any,
        session_id: str | None = None,
        subject_id: SubjectId = SUBJECT_SELF,
    ) -> ThreadId:
        if kind in StimulusRouter._CONVERSATION_KINDS:
            if session_id is None:
                raise ValueError(f"session_id is required for {kind.value} stimuli")
            return ThreadId.from_session(session_id, _source_metadata(payload))
        if kind == StimulusKind.SELF_SIGNAL:
            return ThreadId.cognition(subject_id, _required_goal_id(payload))
        if kind == StimulusKind.CLOCK_TICK:
            return ThreadId.cognition(subject_id, "clock")
        raise ValueError(f"unsupported stimulus kind: {kind.value}")


def _source_metadata(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("source_metadata")
    return metadata if isinstance(metadata, dict) else None


def _required_goal_id(payload: Any) -> str:
    if not isinstance(payload, dict) or not payload.get("goal_id"):
        raise ValueError("self_signal payload must include goal_id")
    return str(payload["goal_id"])
