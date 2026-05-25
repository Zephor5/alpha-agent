"""Detect low-confidence novel claims that immediately formed beliefs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection
from alpha_agent.cognition.reflectors.l1 import AuditContext


class PrematureNovelAutoFormRule:
    name: ClassVar[str] = "premature-novel-auto-form"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        if ctx.interpretation.stance != "novel":
            return
        if not ctx.feedback.formed_belief_ids:
            return
        if not ctx.judgments or min(item.confidence for item in ctx.judgments) >= 0.5:
            return
        for belief_id in ctx.feedback.formed_belief_ids:
            yield ctx.reflection(
                kind=self.name,
                severity="warning",
                target_kind="belief",
                target_id=str(belief_id),
                finding="A low-confidence novel claim was promoted into a belief.",
                suggested_remedy="Require confirmation before forming beliefs from novel claims.",
            )
