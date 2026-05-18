"""Deterministic salience scoring."""

from __future__ import annotations

from alpha_agent.utils.text import contains_any


class SalienceScorer:
    """Score memory importance with transparent first-version heuristics."""

    high_markers = (
        "remember",
        "from now on",
        "important",
        "i prefer",
        "i don't like",
        "do not",
        "don't",
        "always",
        "never",
    )
    correction_markers = ("actually", "correction", "i meant", "not that", "instead")
    decision_markers = ("decided", "decision", "we will", "plan is", "task outcome")

    def score(self, text: str) -> float:
        """Return a salience score between 0 and 1."""

        score = 0.2
        if contains_any(text, self.high_markers):
            score = max(score, 0.85)
        if contains_any(text, self.correction_markers):
            score = max(score, 0.8)
        if contains_any(text, self.decision_markers):
            score = max(score, 0.7)
        if "?" in text and len(text) < 120:
            score = min(score, 0.35)
        if len(text) > 240:
            score = max(score, 0.45)
        return max(0.0, min(1.0, score))
