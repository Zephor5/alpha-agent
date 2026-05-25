"""Deterministic graph snapshot renderer."""

from __future__ import annotations

import re

from alpha_agent.cognition.models import Belief
from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView


class GraphSnapshotRenderer:
    """Render active/recalled beliefs as DOT or Mermaid text."""

    name = "graph_snapshot"

    def __init__(self, *, format: str = "mermaid"):
        if format not in {"mermaid", "dot"}:
            raise ValueError("graph snapshot format must be 'mermaid' or 'dot'")
        self.format = format

    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult:
        beliefs = view.recalled_beliefs[: max(0, budget.max_tokens)]
        payload = (
            self._mermaid(view, beliefs)
            if self.format == "mermaid"
            else self._dot(view, beliefs)
        )
        notes = ["no beliefs available"] if not beliefs else []
        return RenderResult(payload=payload, used_tokens=len(payload) // 4, notes=notes)

    def _mermaid(self, view: CognitionView, beliefs: list[Belief]) -> str:
        lines = ["graph TD", f'  subject["subject:{view.subject.id}"]']
        if view.counterpart is not None:
            lines.append(f'  counterpart["counterpart:{view.counterpart.id}"]')
            lines.append("  subject --> counterpart")
        for belief in beliefs:
            node_id = _node_id(str(belief.id))
            lines.append(f'  {node_id}["{_escape_label(str(belief.content))}"]')
            lines.append(f"  subject --> {node_id}")
            for ref in belief.about:
                target_id = _node_id(f"{ref.kind}:{ref.id}")
                lines.append(f'  {target_id}["{_escape_label(ref.kind + ":" + ref.id)}"]')
                lines.append(f"  {node_id} --> {target_id}")
        return "\n".join(lines)

    def _dot(self, view: CognitionView, beliefs: list[Belief]) -> str:
        lines = ["digraph cognition {", f'  "subject:{view.subject.id}";']
        if view.counterpart is not None:
            lines.append(f'  "counterpart:{view.counterpart.id}";')
            lines.append(f'  "subject:{view.subject.id}" -> "counterpart:{view.counterpart.id}";')
        for belief in beliefs:
            belief_id = f"belief:{belief.id}"
            lines.append(f'  "{belief_id}" [label="{_escape_label(str(belief.content))}"];')
            lines.append(f'  "subject:{view.subject.id}" -> "{belief_id}";')
            for ref in belief.about:
                target = f"{ref.kind}:{ref.id}"
                lines.append(f'  "{target}";')
                lines.append(f'  "{belief_id}" -> "{target}";')
        lines.append("}")
        return "\n".join(lines)


def _node_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
