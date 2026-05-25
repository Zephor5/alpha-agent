"""Deterministic L2 rule set."""

from alpha_agent.cognition.reflectors.l2_rules.feedback_surprise_streak import (
    feedback_surprise_streak,
)
from alpha_agent.cognition.reflectors.l2_rules.lens_shift_flap import lens_shift_flap
from alpha_agent.cognition.reflectors.l2_rules.premature_novel_auto_form_burst import (
    premature_novel_auto_form_burst,
)
from alpha_agent.cognition.reflectors.l2_rules.recurring_contradiction_accepted import (
    recurring_contradiction_accepted,
)

RULES = (
    recurring_contradiction_accepted,
    feedback_surprise_streak,
    lens_shift_flap,
    premature_novel_auto_form_burst,
)

__all__ = [
    "RULES",
    "feedback_surprise_streak",
    "lens_shift_flap",
    "premature_novel_auto_form_burst",
    "recurring_contradiction_accepted",
]
