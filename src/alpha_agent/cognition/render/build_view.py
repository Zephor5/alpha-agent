"""Assemble cognition views from current projections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from alpha_agent.cognition.models import (
    ContextWindow,
    Counterpart,
    Instant,
    Judgment,
    Reference,
    Reflection,
    Situation,
    SituationId,
    ThreadId,
)
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.render.view import CognitionView
from alpha_agent.llm.base import ChatMessage
from alpha_agent.utils.time import utc_now_iso


def build_view(
    *,
    thread_id: ThreadId,
    situation: Situation,
    projections: ProjectionRegistry,
    clock: Callable[[], str] = utc_now_iso,
    window: ContextWindow | None = None,
    counterpart_profile: str | None = None,
    active_judgments: Sequence[Judgment] | None = None,
    matched_procedures: Sequence[Any] | None = None,
    active_strategies: Sequence[Any] | None = None,
    recent_reflections: Sequence[Reflection] | None = None,
    current_query: str | None = None,
    chat_history: Sequence[ChatMessage] | None = None,
) -> CognitionView:
    """Build the renderer-facing view for one thread.

    Current phases have concrete subject, counterpart, context-window,
    procedure, and L1 reflection projections. Later strategy/lens projections
    can feed the optional sequence fields without changing renderer contracts.
    """

    subject = projections.get_typed(SubjectProjection).current()
    if window is None:
        window = projections.get_typed(ContextWindowProjection).get(thread_id, subject)
    counterpart = _counterpart(window, projections)
    reflections = (
        list(recent_reflections)
        if recent_reflections is not None
        else _recent_reflections(projections)
    )
    return CognitionView(
        subject=subject,
        counterpart=counterpart,
        situation=situation,
        window=window,
        counterpart_profile=counterpart_profile,
        active_judgments=list(active_judgments or []),
        matched_procedures=list(matched_procedures or []),
        active_strategies=list(active_strategies or []),
        recent_reflections=reflections,
        assembled_at=Instant(clock()),
        current_query=current_query,
        chat_history=list(chat_history or []),
    )


def situation_from_ref(ref: Reference) -> Situation:
    return Situation(id=SituationId(ref.id))


def _counterpart(window: ContextWindow, projections: ProjectionRegistry) -> Counterpart | None:
    if window.counterpart is None:
        return None
    try:
        return projections.get_typed(CounterpartProjection).get(window.counterpart.id)
    except KeyError:
        return None


def _recent_reflections(projections: ProjectionRegistry) -> list[Reflection]:
    try:
        return projections.get_typed(ReflectionProjection).list_recent(last=5)
    except KeyError:
        return []
