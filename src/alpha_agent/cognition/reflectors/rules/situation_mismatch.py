"""Detect judgments applied outside their declared situation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection
from alpha_agent.cognition.reflectors.l1 import AuditContext

_SITUATION_AGNOSTIC = {"", "*", "any", "all", "reactive_tick"}


class SituationMismatchRule:
    name: ClassVar[str] = "situation-mismatch"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        current = ctx.perception.situation.id
        for item in ctx.judgments:
            applicable = str(item.applicable_under).strip()
            if applicable in _SITUATION_AGNOSTIC or applicable == current:
                continue
            yield ctx.reflection(
                kind=self.name,
                severity="info",
                target_kind="judgment",
                target_id=str(item.id),
                finding="A judgment's applicability does not match the current situation.",
                suggested_remedy="Re-check whether the judgment should apply in this situation.",
            )
