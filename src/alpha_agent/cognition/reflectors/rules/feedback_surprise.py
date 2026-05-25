"""Detect feedback that contradicted the loop's expectations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection
from alpha_agent.cognition.reflectors.l1 import AuditContext


class FeedbackSurpriseRule:
    name: ClassVar[str] = "feedback-surprise"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        if ctx.feedback.matched_expected or not ctx.feedback.surprises:
            return
        yield ctx.reflection(
            kind=self.name,
            severity="info",
            target_kind="loop_run",
            target_id=ctx.tick_id,
            finding="Feedback did not match the expected outcome.",
            suggested_remedy="Review the decision expectation against the actual outcome.",
        )
