"""Deterministic L2 rule set."""

from alpha_agent.cognition.reflectors.l2_rules.lens_shift_flap import lens_shift_flap
from alpha_agent.cognition.reflectors.l2_rules.recurring_contradiction_accepted import (
    recurring_contradiction_accepted,
)

RULES = (
    recurring_contradiction_accepted,
    lens_shift_flap,
)

__all__ = [
    "RULES",
    "lens_shift_flap",
    "recurring_contradiction_accepted",
]
