"""Capability self-assessment from learned procedures."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import Capability, ConfidenceCurve, SubjectRef
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class CapabilitiesAggregator:
    field_name = "capabilities_self_assessed"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> dict[Capability, ConfidenceCurve]:
        del subject, log, window
        try:
            procedures = projections.get_typed(ProcedureProjection).list_active()
        except KeyError:
            return {}
        ranked = sorted(
            procedures,
            key=lambda item: (-float(item.confidence), -int(item.success_count), str(item.id)),
        )
        result: dict[Capability, ConfidenceCurve] = {}
        for procedure in ranked[:12]:
            capability = Capability(str(procedure.trigger))
            result[capability] = ConfidenceCurve(
                "confidence="
                f"{float(procedure.confidence):.3f};success={int(procedure.success_count)};"
                f"failure={int(procedure.failure_count)}"
            )
        return result
