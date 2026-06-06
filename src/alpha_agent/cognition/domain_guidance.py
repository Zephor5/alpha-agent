"""Deterministic routing for LLM-synthesized domain guidance summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from alpha_agent.cognition.models import (
    BeliefScope,
    Reference,
    SummaryBelief,
    SummaryKind,
)
from alpha_agent.cognition.projections.belief import BeliefProjection


@dataclass(frozen=True)
class DomainGuidance:
    """A domain summary routed to one deterministic consumer."""

    belief: SummaryBelief
    target_domain: str


def active_domain_guidance(
    projection: BeliefProjection,
    *,
    target_domain: str,
    counterpart: Reference | None = None,
    now: datetime | None = None,
) -> list[DomainGuidance]:
    """Return active, unexpired domain summaries that apply to one consumer."""

    routed: list[DomainGuidance] = []
    for summary in projection.list_active_summaries(summary_kind=SummaryKind.DOMAIN_SUMMARY):
        domain = summary_target_domain(summary)
        if domain != target_domain:
            continue
        if _is_expired(summary, now=now):
            continue
        if not _scope_applies(summary, counterpart=counterpart):
            continue
        routed.append(DomainGuidance(summary, domain))
    routed.sort(key=lambda item: (str(item.belief.held_since), str(item.belief.id)))
    return routed


def memory_propose_requires_confirmation(
    projection: BeliefProjection,
    *,
    counterpart: Reference | None,
) -> bool:
    """Return whether active memory_propose guidance requires confirmation."""

    return any(
        _requires_confirmation(item.belief)
        for item in active_domain_guidance(
            projection,
            target_domain="memory_propose",
            counterpart=counterpart,
        )
    )


def summary_target_domain(summary: SummaryBelief) -> str | None:
    """Read the optional summary target domain from structured summary metadata."""

    structure = summary.structure if isinstance(summary.structure, dict) else {}
    value = structure.get("target_domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    policy = summary.update_policy if isinstance(summary.update_policy, dict) else {}
    value = policy.get("target_domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _scope_applies(summary: SummaryBelief, *, counterpart: Reference | None) -> bool:
    if summary.scope == BeliefScope.GLOBAL:
        return True
    if summary.scope == BeliefScope.COUNTERPART:
        return counterpart is not None and any(
            ref.kind == counterpart.kind and ref.id == counterpart.id
            for ref in summary.about
        )
    return False


def _is_expired(summary: SummaryBelief, *, now: datetime | None) -> bool:
    valid_until = summary.validity.valid_until
    if valid_until is None:
        return False
    try:
        parsed = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed < (now or datetime.now(UTC))


def _requires_confirmation(summary: SummaryBelief) -> bool:
    structure = summary.structure if isinstance(summary.structure, dict) else {}
    memory_propose = structure.get("memory_propose")
    if isinstance(memory_propose, dict) and _bool_value(
        memory_propose.get("requires_confirmation")
    ):
        return True
    return _bool_value(structure.get("requires_confirmation"))


def _bool_value(value: Any) -> bool:
    return isinstance(value, bool) and value


__all__ = [
    "DomainGuidance",
    "active_domain_guidance",
    "memory_propose_requires_confirmation",
    "summary_target_domain",
]
