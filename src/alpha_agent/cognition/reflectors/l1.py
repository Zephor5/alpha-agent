"""Level-1 deterministic reflector."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from alpha_agent.cognition.models import (
    Counterpart,
    Decision,
    Instant,
    Judgment,
    NLStatement,
    Perception,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
    Subject,
)
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

if TYPE_CHECKING:
    from alpha_agent.cognition.stages.types import (
        AttentionFocus,
        Feedback,
        Interpretation,
        Outcome,
    )
else:
    AttentionFocus = Any
    Feedback = Any
    Interpretation = Any
    Outcome = Any


class ReflectionRule(Protocol):
    """One deterministic L1 reflection rule."""

    name: ClassVar[str]

    def evaluate(self, ctx: AuditContext) -> Iterator[Reflection]:
        """Yield reflections for the current audit context."""


@dataclass(frozen=True)
class AuditContext:
    tick_id: str
    perception: Perception
    focus: AttentionFocus
    interpretation: Interpretation
    judgments: list[Judgment]
    decision: Decision
    outcome: Outcome
    feedback: Feedback
    subject: Subject
    counterpart: Counterpart | None
    projections: ProjectionRegistry
    clock: Callable[[], str] = utc_now_iso
    id_factory: Callable[[], str] = lambda: new_id("reflection")

    def reflection(
        self,
        *,
        kind: str,
        severity: str,
        target_kind: str,
        target_id: str,
        finding: str,
        suggested_remedy: str = "",
    ) -> Reflection:
        return Reflection(
            id=ReflectionId(self.id_factory()),
            level="L1",
            kind=ReflectionKind(kind),
            severity=Severity(severity),
            target=ReflectionTarget(f"{target_kind}:{target_id}"),
            finding=NLStatement(finding),
            suggested_remedy=RemedyHint(suggested_remedy),
            created_at=Instant(self.clock()),
        )


class ReflectorL1:
    """Run a small read-only rule set over one completed reactive tick."""

    def __init__(self, rules: list[ReflectionRule] | None = None):
        self.rules = rules or _default_rules()

    def audit(self, ctx: AuditContext) -> list[Reflection]:
        reflections: list[Reflection] = []
        seen: set[tuple[str, str]] = set()
        for rule in self.rules:
            for reflection in rule.evaluate(ctx):
                key = (str(reflection.kind), str(reflection.target))
                if key in seen:
                    continue
                seen.add(key)
                reflections.append(reflection)
        return reflections


def _default_rules() -> list[ReflectionRule]:
    from alpha_agent.cognition.reflectors.rules.contradiction_accepted import (
        ContradictionAcceptedRule,
    )
    from alpha_agent.cognition.reflectors.rules.feedback_surprise import FeedbackSurpriseRule
    from alpha_agent.cognition.reflectors.rules.low_confidence_high_stakes import (
        LowConfidenceHighStakesRule,
    )
    from alpha_agent.cognition.reflectors.rules.premature_novel_auto_form import (
        PrematureNovelAutoFormRule,
    )
    from alpha_agent.cognition.reflectors.rules.situation_mismatch import SituationMismatchRule
    from alpha_agent.cognition.reflectors.rules.unsupported_tool_call import (
        UnsupportedToolCallRule,
    )

    return [
        LowConfidenceHighStakesRule(),
        ContradictionAcceptedRule(),
        SituationMismatchRule(),
        UnsupportedToolCallRule(),
        PrematureNovelAutoFormRule(),
        FeedbackSurpriseRule(),
    ]
