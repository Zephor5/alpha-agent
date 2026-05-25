"""Detect high-stakes judgments made with weak confidence."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection, ValueKind
from alpha_agent.cognition.reflectors.l1 import AuditContext


class LowConfidenceHighStakesRule:
    name: ClassVar[str] = "low-confidence-high-stakes"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        for item in ctx.judgments:
            if item.confidence >= 0.4:
                continue
            if _high_stakes_weight(item.value_weights) <= 0.7:
                continue
            yield ctx.reflection(
                kind=self.name,
                severity="warning",
                target_kind="judgment",
                target_id=str(item.id),
                finding="A high-stakes judgment was accepted with low confidence.",
                suggested_remedy="Ask for stronger evidence before relying on this judgment.",
            )


def _high_stakes_weight(weights: dict[object, float]) -> float:
    candidates = ("existence", "safety", ValueKind.SAFETY)
    values = []
    for key in candidates:
        if key in weights:
            values.append(float(weights[key]))
    return max(values, default=0.0)
