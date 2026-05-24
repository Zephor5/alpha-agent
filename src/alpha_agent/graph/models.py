"""Auxiliary graph domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.memory.models import ConversationMessage, SemanticMemory


@dataclass(frozen=True)
class EntityNode:
    """Loose entity node used as an auxiliary memory index."""

    id: str
    name: str
    kind: str | None
    aliases: list[str]
    salience: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationEdge:
    """Loose relation between two entity nodes."""

    id: str
    source_node_id: str
    target_node_id: str
    relation_type: str = "related_to"
    evidence_memory_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationEdgeAudit:
    """Relation edge plus active evidence memories and transcript sources."""

    edge: RelationEdge
    source_node: EntityNode
    target_node: EntityNode
    evidence_memories: list[SemanticMemory]
    source_messages: list[ConversationMessage]
