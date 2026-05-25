"""Stable preference references from high-confidence value beliefs."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import BeliefRef, CognitiveType, SubjectRef, belief_ref
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class StablePreferencesAggregator:
    field_name = "stable_preferences"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> list[BeliefRef]:
        del subject, log, window
        try:
            beliefs = projections.get_typed(BeliefProjection).list_active()
        except KeyError:
            return []
        stable = [
            belief
            for belief in beliefs
            if belief.cognitive_type == CognitiveType.VALUE
            and float(belief.confidence) >= 0.8
            and not belief.about
        ]
        return [
            belief_ref(item.id)
            for item in sorted(
                stable,
                key=lambda belief: (-float(belief.confidence), str(belief.id)),
            )
        ][:12]
