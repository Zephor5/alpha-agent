"""Load and rank procedural skill definitions.

The manager deliberately keeps skill loading simple: builtin skills are Markdown
files with a small metadata header, and the returned dataclass can later be
converted into a procedural-memory storage record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_BUILTIN_DIR = Path(__file__).with_name("builtin")
_FRONTMATTER_BOUNDARY = "---"
_WORD_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class SkillDefinition:
    """A reusable procedural skill loaded from Markdown."""

    id: str
    name: str
    description: str
    trigger: str
    procedure_markdown: str
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_procedural_memory_dict(self) -> dict[str, Any]:
        """Return values shaped like the planned ProceduralMemory model."""
        now = datetime.now(UTC).isoformat()
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "procedure_markdown": self.procedure_markdown,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "created_at": now,
            "updated_at": now,
            "metadata": dict(self.metadata),
        }


class SkillManager:
    """Load builtin procedural skills and rank them by lexical relevance."""

    def __init__(self, builtin_dir: Path | None = None) -> None:
        self.builtin_dir = builtin_dir or _BUILTIN_DIR

    def load_builtin_skills(self) -> list[SkillDefinition]:
        """Load all builtin Markdown skills from the configured directory."""
        if not self.builtin_dir.exists():
            return []

        skills = [
            self.load_markdown_skill(path)
            for path in sorted(self.builtin_dir.glob("*.md"))
            if path.is_file()
        ]
        return skills

    def load_markdown_skill(self, path: Path) -> SkillDefinition:
        """Load one skill definition from a Markdown file."""
        raw_markdown = path.read_text(encoding="utf-8")
        metadata, body = _split_frontmatter(raw_markdown)
        skill_id = metadata.get("id") or f"builtin:{path.stem}"
        name = metadata.get("name") or _title_from_stem(path.stem)
        description = metadata.get("description") or ""
        trigger = metadata.get("trigger") or name
        confidence = _parse_confidence(metadata.get("confidence"))

        return SkillDefinition(
            id=skill_id,
            name=name,
            description=description,
            trigger=trigger,
            procedure_markdown=body.strip(),
            confidence=confidence,
            metadata={
                "source": "builtin",
                "path": str(path),
                "frontmatter": dict(metadata),
            },
        )

    def get_builtin_skill(self, name_or_id: str) -> SkillDefinition | None:
        """Return a builtin skill by id, name, or filename stem."""
        normalized = _normalize_lookup(name_or_id)
        for skill in self.load_builtin_skills():
            candidates = {
                _normalize_lookup(skill.id),
                _normalize_lookup(skill.name),
                _normalize_lookup(skill.id.removeprefix("builtin:")),
            }
            if normalized in candidates:
                return skill
        return None

    def find_relevant_skills(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[SkillDefinition]:
        """Return builtin skills ranked by simple keyword overlap."""
        if limit <= 0:
            return []

        query_terms = _tokenize(query)
        ranked: list[tuple[float, str, SkillDefinition]] = []
        for skill in self.load_builtin_skills():
            score = _score_skill(skill, query_terms)
            if score > 0:
                ranked.append((score, skill.name.lower(), skill))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [skill for _, _, skill in ranked[:limit]]


def _split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}, markdown

    metadata: dict[str, str] = {}
    body_start = 0
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_BOUNDARY:
            body_start = index + 1
            break

        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"')
    else:
        return {}, markdown

    return metadata, "\n".join(lines[body_start:])


def _parse_confidence(value: str | None) -> float:
    if value is None:
        return 0.75
    try:
        confidence = float(value)
    except ValueError:
        return 0.75
    return max(0.0, min(1.0, confidence))


def _score_skill(skill: SkillDefinition, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0

    name_terms = _tokenize(skill.name)
    trigger_terms = _tokenize(skill.trigger)
    description_terms = _tokenize(skill.description)
    body_terms = _tokenize(skill.procedure_markdown)

    return (
        len(query_terms & name_terms) * 3.0
        + len(query_terms & trigger_terms) * 2.5
        + len(query_terms & description_terms) * 1.5
        + len(query_terms & body_terms) * 0.5
    )


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _normalize_lookup(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _title_from_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").title()
