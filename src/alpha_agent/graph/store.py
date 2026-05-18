"""Optional SQLite graph store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from alpha_agent.graph.models import EntityNode, RelationEdge
from alpha_agent.utils.ids import new_id


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    loaded = json.loads(value)
    return [str(item) for item in loaded] if isinstance(loaded, list) else []


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


class GraphStore:
    """Small optional store for entity nodes and relation edges."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def connect(self) -> sqlite3.Connection:
        """Open a connection to the existing Alpha Agent database."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert_node(
        self,
        name: str,
        kind: str | None = None,
        aliases: list[str] | None = None,
        salience: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> EntityNode:
        """Insert or update an entity node by name."""

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO entity_nodes (id, name, kind, aliases, salience, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    kind = COALESCE(excluded.kind, entity_nodes.kind),
                    aliases = excluded.aliases,
                    salience = max(entity_nodes.salience, excluded.salience),
                    metadata = excluded.metadata
                """,
                (
                    new_id("node"),
                    name,
                    kind,
                    _dumps(aliases or []),
                    max(0.0, min(1.0, salience)),
                    _dumps(metadata or {}),
                ),
            )
            row = conn.execute("SELECT * FROM entity_nodes WHERE name = ?", (name,)).fetchone()
        return self._node_from_row(row)

    def list_nodes(self, limit: int = 100) -> list[EntityNode]:
        """List entity nodes ordered by salience."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entity_nodes ORDER BY salience DESC, name ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: str = "related_to",
        evidence_memory_ids: list[str] | None = None,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> RelationEdge:
        """Insert a relation edge."""

        edge = RelationEdge(
            id=new_id("edge"),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            evidence_memory_ids=evidence_memory_ids or [],
            confidence=max(0.0, min(1.0, confidence)),
            metadata=metadata or {},
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO relation_edges
                    (id, source_node_id, target_node_id, relation_type,
                     evidence_memory_ids, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.relation_type,
                    _dumps(edge.evidence_memory_ids),
                    edge.confidence,
                    _dumps(edge.metadata),
                ),
            )
        return edge

    def list_edges(self, limit: int = 100) -> list[RelationEdge]:
        """List relation edges."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM relation_edges ORDER BY confidence DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def _node_from_row(self, row: sqlite3.Row) -> EntityNode:
        return EntityNode(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            aliases=_loads_list(row["aliases"]),
            salience=float(row["salience"]),
            metadata=_loads_dict(row["metadata"]),
        )

    def _edge_from_row(self, row: sqlite3.Row) -> RelationEdge:
        return RelationEdge(
            id=row["id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            relation_type=row["relation_type"],
            evidence_memory_ids=_loads_list(row["evidence_memory_ids"]),
            confidence=float(row["confidence"]),
            metadata=_loads_dict(row["metadata"]),
        )
