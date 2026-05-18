"""Working memory manager."""

from __future__ import annotations

from alpha_agent.memory.models import WorkingMemoryItem
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class WorkingMemoryManager:
    """Manage short-lived active context for a session."""

    def __init__(self, store: MemoryStore, limit: int = 12):
        self.store = store
        self.limit = limit

    def add_active_context(
        self,
        session_id: str,
        content: str,
        source_event_id: str | None = None,
        priority: float = 0.5,
    ) -> WorkingMemoryItem:
        """Add an active context item and enforce the session bound."""

        item = WorkingMemoryItem(
            id=new_id("wm"),
            session_id=session_id,
            content=content,
            source_event_id=source_event_id,
            priority=max(0.0, min(1.0, priority)),
            expires_at=None,
            created_at=utc_now_iso(),
            metadata={},
        )
        self.store.add_working_memory(item)
        self.expire_old_items(session_id)
        return item

    def expire_old_items(self, session_id: str) -> None:
        """Remove expired and overflow working memory items."""

        self.store.expire_working_memory(session_id, self.limit)

    def get_active_context(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[WorkingMemoryItem]:
        """Return the highest-priority active context for a session."""

        return self.store.list_working_memory(session_id, limit or self.limit)
