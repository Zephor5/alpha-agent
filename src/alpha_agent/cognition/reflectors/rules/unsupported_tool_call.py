"""Detect tool-use decisions without a supporting tool-use judgment."""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from alpha_agent.cognition.models import Reflection
from alpha_agent.cognition.reflectors.l1 import AuditContext


class UnsupportedToolCallRule:
    name: ClassVar[str] = "unsupported-tool-call"

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        if str(ctx.decision.action) != "use_tool":
            return
        if any(_requires_tool(item.claim) for item in ctx.judgments):
            return
        yield ctx.reflection(
            kind=self.name,
            severity="warning",
            target_kind="decision",
            target_id=str(ctx.decision.id),
            finding="A tool-use action was selected without any judgment requiring a tool.",
            suggested_remedy="Only call tools when a judgment explicitly calls for tool use.",
        )


def _requires_tool(claim: object) -> bool:
    text = str(claim).casefold()
    return "tool" in text or "use_tool" in text
