"""Deterministic conflict resolver shaped by the subject ValueLens."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.cognition.models import Belief, BeliefId, ValueKind
from alpha_agent.cognition.models.value import ValueLens
from alpha_agent.cognition.value.lens import normalize_lens

TIE_EPSILON = 1e-9


@dataclass(frozen=True)
class ConflictResolution:
    winner_id: BeliefId | None
    loser_id: BeliefId | None
    tie: bool
    rationale: str
    by_lens_priority: list[ValueKind]
    margin: float

    def to_payload(self) -> dict[str, object]:
        return {
            "winner_id": str(self.winner_id) if self.winner_id is not None else None,
            "loser_id": str(self.loser_id) if self.loser_id is not None else None,
            "tie": self.tie,
            "rationale": self.rationale,
            "decisive_value_kinds": [value.value for value in self.by_lens_priority],
            "margin": self.margin,
        }


def resolve_conflict(left: Belief, right: Belief, lens: ValueLens) -> ConflictResolution:
    normalized = normalize_lens(lens)
    left_score = _score(left, normalized)
    right_score = _score(right, normalized)
    margin = round(abs(left_score - right_score), 6)
    decisive = _decisive_values(left, right, normalized)
    if abs(left_score - right_score) <= TIE_EPSILON:
        return ConflictResolution(
            winner_id=None,
            loser_id=None,
            tie=True,
            rationale=(
                "tie under current value lens "
                f"(left={left_score:.3f}, right={right_score:.3f})"
            ),
            by_lens_priority=decisive,
            margin=0.0,
        )
    winner = left if left_score > right_score else right
    loser = right if left_score > right_score else left
    return ConflictResolution(
        winner_id=winner.id,
        loser_id=loser.id,
        tie=False,
        rationale=(
            "winner selected by current value lens "
            f"(winner={max(left_score, right_score):.3f}, loser={min(left_score, right_score):.3f})"
        ),
        by_lens_priority=decisive,
        margin=margin,
    )


def _score(belief: Belief, lens: ValueLens) -> float:
    total = 0.0
    for value in lens.priorities:
        profile_weight = float(belief.value_profile.weights.get(value, 0.0))
        priority_weight = float(lens.weights.get(value, 1.0))
        sensitivity = float(lens.sensitivity.get(value, 1.0))
        total += profile_weight * priority_weight * sensitivity
    return total


def _decisive_values(left: Belief, right: Belief, lens: ValueLens) -> list[ValueKind]:
    values: list[ValueKind] = []
    for value in lens.priorities:
        left_part = float(left.value_profile.weights.get(value, 0.0)) * float(
            lens.weights.get(value, 1.0)
        ) * float(lens.sensitivity.get(value, 1.0))
        right_part = float(right.value_profile.weights.get(value, 0.0)) * float(
            lens.weights.get(value, 1.0)
        ) * float(lens.sensitivity.get(value, 1.0))
        if abs(left_part - right_part) > TIE_EPSILON:
            values.append(value)
    return values
