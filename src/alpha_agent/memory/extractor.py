"""Deterministic post-turn memory extraction."""

from __future__ import annotations

import re

from alpha_agent.memory.models import ExtractedMemoryCandidate
from alpha_agent.memory.salience import SalienceScorer


class MemoryExtractor:
    """Extract transparent candidate memories without an LLM."""

    preference_patterns: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bi prefer (?P<object>.+)", re.IGNORECASE), "prefers"),
        (re.compile(r"\bi like (?P<object>.+)", re.IGNORECASE), "likes"),
        (re.compile(r"\bi don't like (?P<object>.+)", re.IGNORECASE), "dislikes"),
    )
    durable_fact_patterns = (
        re.compile(r"\bmy (?P<subject>[a-zA-Z][\w -]{1,40}) is (?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bi am (?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bi work (?:at|for) (?P<object>.+)", re.IGNORECASE),
    )
    procedural_patterns = (
        re.compile(r"\bwhen i ask (?:you )?to (?P<trigger>.+), (?P<procedure>.+)", re.IGNORECASE),
        re.compile(r"\bfrom now on,? (?P<procedure>.+)", re.IGNORECASE),
    )

    def __init__(self, salience_scorer: SalienceScorer | None = None):
        self.salience_scorer = salience_scorer or SalienceScorer()

    def extract(
        self,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
    ) -> list[ExtractedMemoryCandidate]:
        """Extract candidate memories from a completed turn."""

        del assistant_response
        candidates: list[ExtractedMemoryCandidate] = []
        salience = self.salience_scorer.score(user_message)

        for pattern, predicate in self.preference_patterns:
            match = pattern.search(user_message)
            if match:
                value = self._clean_value(match.group("object"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="semantic",
                        content=f"User {predicate}: {value}",
                        subject="user",
                        predicate=predicate,
                        object=value,
                        salience=max(salience, 0.85),
                        confidence=0.7,
                        source_event_ids=source_event_ids,
                        metadata={"extractor": "preference_pattern"},
                    )
                )
                break

        for pattern in self.durable_fact_patterns:
            match = pattern.search(user_message)
            if match:
                if "subject" in match.groupdict():
                    subject = f"user.{self._clean_value(match.group('subject')).replace(' ', '_')}"
                    predicate = "is"
                    value = self._clean_value(match.group("object"))
                elif "work" in pattern.pattern:
                    subject = "user"
                    predicate = "works_at"
                    value = self._clean_value(match.group("object"))
                else:
                    subject = "user"
                    predicate = "is"
                    value = self._clean_value(match.group("object"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="semantic",
                        content=f"{subject} {predicate} {value}",
                        subject=subject,
                        predicate=predicate,
                        object=value,
                        salience=max(salience, 0.65),
                        confidence=0.65,
                        source_event_ids=source_event_ids,
                        metadata={"extractor": "durable_fact_pattern"},
                    )
                )
                break

        lowered = user_message.lower()
        if "remember" in lowered or "important" in lowered or "actually" in lowered:
            candidates.append(
                ExtractedMemoryCandidate(
                    type="episodic",
                    content=f"User said: {user_message}",
                    salience=max(salience, 0.75),
                    confidence=0.7,
                    source_event_ids=source_event_ids,
                    metadata={"extractor": "explicit_or_correction"},
                )
            )

        for pattern in self.procedural_patterns:
            match = pattern.search(user_message)
            if match:
                procedure = self._clean_value(match.groupdict().get("procedure", user_message))
                trigger = self._clean_value(match.groupdict().get("trigger", "user instruction"))
                candidates.append(
                    ExtractedMemoryCandidate(
                        type="procedural_candidate",
                        content=procedure,
                        subject="user",
                        predicate="procedure",
                        object=trigger,
                        salience=max(salience, 0.75),
                        confidence=0.55,
                        source_event_ids=source_event_ids,
                        metadata={"extractor": "procedural_pattern", "trigger": trigger},
                    )
                )
                break

        return candidates

    def _clean_value(self, value: str) -> str:
        return value.strip().rstrip(".!").strip()
