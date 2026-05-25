"""Detect judgments that accept one belief as both support and contradiction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection
from alpha_agent.cognition.reflectors.l1 import AuditContext


class ContradictionAcceptedRule:
    name: ClassVar[str] = "contradiction-accepted"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        for item in ctx.judgments:
            supported = {(ref.kind, ref.id) for ref in item.supports}
            undermined = {(ref.kind, ref.id) for ref in item.undermined_by}
            overlap = supported & undermined
            if not overlap:
                continue
            yield ctx.reflection(
                kind=self.name,
                severity="blocker",
                target_kind="judgment",
                target_id=str(item.id),
                finding="A judgment accepted the same belief as support and contradiction.",
                suggested_remedy=(
                    "Resolve the contradiction before treating the judgment as stable."
                ),
            )
