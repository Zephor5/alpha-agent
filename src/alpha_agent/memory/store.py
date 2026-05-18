"""SQLite persistence for Alpha Agent memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from alpha_agent.memory.models import (
    EpisodicMemory,
    Event,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemoryItem,
)
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

T = TypeVar("T")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


class MemoryStore:
    """Low-level SQLite operations for events and memory layers."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.fts_available = False

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        """Create database tables and optional FTS5 indexes."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            self.fts_available = self._try_initialize_fts(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run operations in a SQLite transaction."""

        with self.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _try_initialize_fts(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
                USING fts5(id UNINDEXED, content);
                CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(memory_id UNINDEXED, content, summary);
                CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts
                USING fts5(memory_id UNINDEXED, subject, predicate, object, content);
                CREATE VIRTUAL TABLE IF NOT EXISTS procedural_fts
                USING fts5(memory_id UNINDEXED, name, description, trigger, procedure_markdown);
                """
            )
        except sqlite3.OperationalError:
            return False
        return True

    def _has_fts_table(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _with_conn(
        self,
        conn: sqlite3.Connection | None,
        fn: Callable[[sqlite3.Connection], T],
    ) -> T:
        if conn is not None:
            return fn(conn)
        with self.connect() as local:
            self.fts_available = self._has_fts_table(local, "episodic_fts")
            return fn(local)

    def insert_event(
        self,
        event: Event,
        conn: sqlite3.Connection | None = None,
    ) -> Event:
        """Insert an event and update optional FTS index."""

        def op(db: sqlite3.Connection) -> Event:
            db.execute(
                """
                INSERT INTO events (id, session_id, role, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.role,
                    event.content,
                    event.created_at,
                    _dumps(event.metadata),
                ),
            )
            if self._has_fts_table(db, "events_fts"):
                db.execute(
                    "INSERT INTO events_fts (id, content) VALUES (?, ?)",
                    (event.id, event.content),
                )
            return event

        return self._with_conn(conn, op)

    def list_events(self, session_id: str | None = None, limit: int = 50) -> list[Event]:
        """List recent events, optionally scoped to a session."""

        with self.connect() as conn:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT * FROM events WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def add_working_memory(
        self,
        item: WorkingMemoryItem,
        conn: sqlite3.Connection | None = None,
    ) -> WorkingMemoryItem:
        """Insert a working memory item."""

        def op(db: sqlite3.Connection) -> WorkingMemoryItem:
            db.execute(
                """
                INSERT INTO working_memory
                    (id, session_id, content, source_event_id, priority,
                     expires_at, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.session_id,
                    item.content,
                    item.source_event_id,
                    item.priority,
                    item.expires_at,
                    item.created_at,
                    _dumps(item.metadata),
                ),
            )
            return item

        return self._with_conn(conn, op)

    def list_working_memory(self, session_id: str, limit: int = 12) -> list[WorkingMemoryItem]:
        """Return active working memory ordered by priority and recency."""

        now = utc_now_iso()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM working_memory
                WHERE session_id = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY priority DESC, created_at DESC
                LIMIT ?
                """,
                (session_id, now, limit),
            ).fetchall()
        return [self._working_from_row(row) for row in rows]

    def expire_working_memory(self, session_id: str, max_items: int) -> None:
        """Remove expired items and keep only the highest-priority recent active items."""

        now = utc_now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                DELETE FROM working_memory
                WHERE session_id = ? AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (session_id, now),
            )
            conn.execute(
                """
                DELETE FROM working_memory
                WHERE id IN (
                    SELECT id FROM working_memory
                    WHERE session_id = ?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (session_id, max_items),
            )

    def prune_low_priority_working_memory(self, priority_below: float = 0.25) -> int:
        """Delete low-priority working memory items and return the number removed."""

        with self.transaction() as conn:
            before = int(conn.execute("SELECT count(*) FROM working_memory").fetchone()[0])
            conn.execute(
                "DELETE FROM working_memory WHERE priority < ?",
                (priority_below,),
            )
            after = int(conn.execute("SELECT count(*) FROM working_memory").fetchone()[0])
        return before - after

    def insert_episodic_memory(
        self,
        memory: EpisodicMemory,
        conn: sqlite3.Connection | None = None,
    ) -> EpisodicMemory:
        """Insert an episodic memory and optional FTS row."""

        def op(db: sqlite3.Connection) -> EpisodicMemory:
            db.execute(
                """
                INSERT INTO episodic_memories
                    (id, content, summary, source_event_ids, people, places, topics,
                     salience, confidence, created_at, last_accessed_at, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.summary,
                    _dumps(memory.source_event_ids),
                    _dumps(memory.people),
                    _dumps(memory.places),
                    _dumps(memory.topics),
                    memory.salience,
                    memory.confidence,
                    memory.created_at,
                    memory.last_accessed_at,
                    memory.access_count,
                    _dumps(memory.metadata),
                ),
            )
            if self._has_fts_table(db, "episodic_fts"):
                db.execute(
                    "INSERT INTO episodic_fts (memory_id, content, summary) VALUES (?, ?, ?)",
                    (memory.id, memory.content, memory.summary),
                )
            return memory

        return self._with_conn(conn, op)

    def list_episodic_memories(self, limit: int = 50) -> list[EpisodicMemory]:
        """List recent episodic memories."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def search_episodic(self, query: str, limit: int = 20) -> list[EpisodicMemory]:
        """Search episodic memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "episodic_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM episodic_fts f
                    JOIN episodic_memories m ON m.id = f.memory_id
                    WHERE episodic_fts MATCH ?
                    ORDER BY bm25(episodic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM episodic_memories
                    WHERE content LIKE ? OR summary LIKE ?
                    ORDER BY salience DESC, created_at DESC LIMIT ?
                    """,
                    (like, like, limit),
                ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def upsert_semantic_memory(
        self,
        memory: SemanticMemory,
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory:
        """Insert or update a semantic fact keyed by subject, predicate, and object."""

        def op(db: sqlite3.Connection) -> SemanticMemory:
            db.execute(
                """
                INSERT INTO semantic_memories
                    (id, subject, predicate, object, content, confidence, salience,
                     source_memory_ids, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject, predicate, object) DO UPDATE SET
                    content = excluded.content,
                    confidence = max(semantic_memories.confidence, excluded.confidence),
                    salience = max(semantic_memories.salience, excluded.salience),
                    source_memory_ids = excluded.source_memory_ids,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    memory.subject.lower().strip(),
                    memory.predicate.lower().strip(),
                    memory.object.lower().strip(),
                    memory.content,
                    memory.confidence,
                    memory.salience,
                    _dumps(memory.source_memory_ids),
                    memory.created_at,
                    memory.updated_at,
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                """
                SELECT * FROM semantic_memories
                WHERE subject = ? AND predicate = ? AND object = ?
                """,
                (
                    memory.subject.lower().strip(),
                    memory.predicate.lower().strip(),
                    memory.object.lower().strip(),
                ),
            ).fetchone()
            saved = self._semantic_from_row(row)
            if self._has_fts_table(db, "semantic_fts"):
                db.execute("DELETE FROM semantic_fts WHERE memory_id = ?", (saved.id,))
                db.execute(
                    """
                    INSERT INTO semantic_fts
                        (memory_id, subject, predicate, object, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (saved.id, saved.subject, saved.predicate, saved.object, saved.content),
                )
            return saved

        return self._with_conn(conn, op)

    def list_semantic_memories(self, limit: int = 50) -> list[SemanticMemory]:
        """List semantic memories ordered by salience and recency."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_memories ORDER BY salience DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._semantic_from_row(row) for row in rows]

    def search_semantic(self, query: str, limit: int = 20) -> list[SemanticMemory]:
        """Search semantic memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "semantic_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM semantic_fts f
                    JOIN semantic_memories m ON m.id = f.memory_id
                    WHERE semantic_fts MATCH ?
                    ORDER BY bm25(semantic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM semantic_memories
                    WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR content LIKE ?
                    ORDER BY salience DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
        return [self._semantic_from_row(row) for row in rows]

    def upsert_procedural_memory(
        self,
        memory: ProceduralMemory,
        conn: sqlite3.Connection | None = None,
    ) -> ProceduralMemory:
        """Insert or update a procedural memory keyed by name."""

        def op(db: sqlite3.Connection) -> ProceduralMemory:
            db.execute(
                """
                INSERT INTO procedural_memories
                    (id, name, description, trigger, procedure_markdown, success_count,
                     failure_count, confidence, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    trigger = excluded.trigger,
                    procedure_markdown = excluded.procedure_markdown,
                    confidence = max(procedural_memories.confidence, excluded.confidence),
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    memory.name,
                    memory.description,
                    memory.trigger,
                    memory.procedure_markdown,
                    memory.success_count,
                    memory.failure_count,
                    memory.confidence,
                    memory.created_at,
                    memory.updated_at,
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                "SELECT * FROM procedural_memories WHERE name = ?",
                (memory.name,),
            ).fetchone()
            saved = self._procedural_from_row(row)
            if self._has_fts_table(db, "procedural_fts"):
                db.execute("DELETE FROM procedural_fts WHERE memory_id = ?", (saved.id,))
                db.execute(
                    """
                    INSERT INTO procedural_fts
                        (memory_id, name, description, trigger, procedure_markdown)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        saved.id,
                        saved.name,
                        saved.description,
                        saved.trigger,
                        saved.procedure_markdown,
                    ),
                )
            return saved

        return self._with_conn(conn, op)

    def list_procedural_memories(self, limit: int = 50) -> list[ProceduralMemory]:
        """List procedural memories."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM procedural_memories
                ORDER BY confidence DESC, updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def search_procedural(self, query: str, limit: int = 20) -> list[ProceduralMemory]:
        """Search procedural memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "procedural_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM procedural_fts f
                    JOIN procedural_memories m ON m.id = f.memory_id
                    WHERE procedural_fts MATCH ?
                    ORDER BY bm25(procedural_fts), m.confidence DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM procedural_memories
                    WHERE name LIKE ?
                       OR description LIKE ?
                       OR trigger LIKE ?
                       OR procedure_markdown LIKE ?
                    ORDER BY confidence DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def log_memory_access(
        self,
        memory_id: str,
        memory_type: str,
        query: str,
        score: float,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record that a memory was retrieved and update access counters when applicable."""

        def op(db: sqlite3.Connection) -> None:
            db.execute(
                """
                INSERT INTO memory_access_log
                    (id, memory_id, memory_type, query, accessed_at, score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id("access"), memory_id, memory_type, query, utc_now_iso(), score, "{}"),
            )
            if memory_type == "episodic":
                db.execute(
                    """
                    UPDATE episodic_memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (utc_now_iso(), memory_id),
                )

        self._with_conn(conn, op)

    def stats(self) -> dict[str, int]:
        """Return counts by memory table."""

        tables = {
            "events": "events",
            "working_memory": "working_memory",
            "episodic": "episodic_memories",
            "semantic": "semantic_memories",
            "procedural": "procedural_memories",
            "entity_nodes": "entity_nodes",
            "relation_edges": "relation_edges",
        }
        with self.connect() as conn:
            return {
                key: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for key, table in tables.items()
            }

    def _fts_query(self, query: str) -> str:
        terms = [term.replace('"', "") for term in query.split() if term.strip()]
        return " OR ".join(f'"{term}"' for term in terms) or '""'

    def _event_from_row(self, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _working_from_row(self, row: sqlite3.Row) -> WorkingMemoryItem:
        return WorkingMemoryItem(
            id=row["id"],
            session_id=row["session_id"],
            content=row["content"],
            source_event_id=row["source_event_id"],
            priority=float(row["priority"]),
            expires_at=row["expires_at"],
            created_at=row["created_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _episodic_from_row(self, row: sqlite3.Row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            source_event_ids=_loads_list(row["source_event_ids"]),
            people=_loads_list(row["people"]),
            places=_loads_list(row["places"]),
            topics=_loads_list(row["topics"]),
            salience=float(row["salience"]),
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            metadata=_loads_dict(row["metadata"]),
        )

    def _semantic_from_row(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            content=row["content"],
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            source_memory_ids=_loads_list(row["source_memory_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _procedural_from_row(self, row: sqlite3.Row) -> ProceduralMemory:
        return ProceduralMemory(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            trigger=row["trigger"],
            procedure_markdown=row["procedure_markdown"],
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
        )
