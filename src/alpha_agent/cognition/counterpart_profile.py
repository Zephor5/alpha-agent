"""Counterpart profile helpers backed by digest beliefs."""

from __future__ import annotations

from collections.abc import Sequence

from alpha_agent.cognition.models import Belief, Reference
from alpha_agent.cognition.projections.belief import BeliefProjection

COUNTERPART_DIGEST_OBJECT_PREFIX = "counterpart_digest:"


def counterpart_digest_object(counterpart_id: str) -> str:
    return f"{COUNTERPART_DIGEST_OBJECT_PREFIX}{counterpart_id}"


def active_counterpart_digest(
    projection: BeliefProjection,
    counterpart: Reference,
) -> Belief | None:
    return latest_counterpart_digest(projection.recall_about(counterpart), counterpart.id)


def latest_counterpart_digest(
    beliefs: Sequence[Belief],
    counterpart_id: str,
) -> Belief | None:
    object_id = counterpart_digest_object(counterpart_id)
    digests = [belief for belief in beliefs if str(belief.object) == object_id]
    return max(digests, key=lambda item: str(item.held_since)) if digests else None
