"""Procedural memory manager."""

from __future__ import annotations

from alpha_agent.memory.models import ProceduralMemory
from alpha_agent.memory.store import MemoryStore
from alpha_agent.skills.manager import SkillManager
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class ProceduralMemoryManager:
    """Manage reusable procedures and builtin skills."""

    def __init__(self, store: MemoryStore, skill_manager: SkillManager | None = None):
        self.store = store
        self.skill_manager = skill_manager or SkillManager()

    def load_builtin_skills(self) -> list[ProceduralMemory]:
        """Load bundled markdown skills into procedural memory."""

        memories: list[ProceduralMemory] = []
        for skill in self.skill_manager.load_builtin_skills():
            now = utc_now_iso()
            memories.append(
                self.store.upsert_procedural_memory(
                    ProceduralMemory(
                        id=new_id("proc"),
                        name=skill.name,
                        description=skill.description,
                        trigger=skill.trigger,
                        procedure_markdown=skill.procedure_markdown,
                        success_count=0,
                        failure_count=0,
                        confidence=0.75,
                        created_at=now,
                        updated_at=now,
                        metadata={"builtin": True},
                    )
                )
            )
        return memories

    def retrieve_relevant(self, query: str, limit: int = 5) -> list[ProceduralMemory]:
        """Retrieve relevant procedural memories."""

        return self.store.search_procedural(query, limit)
