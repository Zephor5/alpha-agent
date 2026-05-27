"""Daemon-owned agent factory and per-session cache."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import LLMProvider
from alpha_agent.llm.codex import CodexResponsesProvider
from alpha_agent.llm.deepseek import DeepSeekProvider
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.llm.openai_compatible import OpenAICompatibleProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore


@dataclass(slots=True)
class ManagedAgent:
    """Cached daemon agent entry."""

    agent: AlphaAgent
    last_used_at: float


def build_provider(config: AlphaConfig) -> LLMProvider:
    """Build the configured LLM provider for daemon-owned agents."""

    if config.llm_provider in {"mock", ""}:
        return MockLLMProvider()
    if config.llm_provider in {"openai", "openai-compatible", "compatible"}:
        return OpenAICompatibleProvider(config)
    if config.llm_provider in {"deepseek"}:
        return DeepSeekProvider(config)
    if config.llm_provider in {"codex", "openai-codex", "openai_codex"}:
        return CodexResponsesProvider(config)
    raise ValueError(f"Unknown ALPHA_LLM_PROVIDER: {config.llm_provider}")


def initialize_store(config: AlphaConfig) -> StateStore:
    """Initialize and return the daemon-owned state store."""

    store = StateStore(config.db_path)
    store.initialize()
    return store


class AgentFactory:
    """Build AlphaAgent instances sharing daemon-owned infrastructure."""

    def __init__(self, config: AlphaConfig, store: StateStore):
        self.config = config
        self.store = store
        self._lock = Lock()

    def create(self) -> AlphaAgent:
        """Create one session-scoped AlphaAgent."""

        with self._lock:
            provider = build_provider(self.config)
        return AlphaAgent(
            store=self.store,
            llm_provider=provider,
            llm_debug_logging=self.config.llm_debug_logging,
            llm_trace_log_path=Path(self.config.log_dir) / "llm.jsonl",
            llm_context_config=self.config.llm_context,
            max_context_tokens=self.config.max_context_tokens_for_provider(
                self.config.llm_provider
            ),
        )


class AgentManager:
    """Cache one AlphaAgent per active session id."""

    def __init__(
        self,
        factory: AgentFactory,
        *,
        idle_ttl_seconds: float = 3600.0,
        max_size: int = 128,
    ):
        self.factory = factory
        self.idle_ttl_seconds = idle_ttl_seconds
        self.max_size = max(1, max_size)
        self._agents: OrderedDict[str, ManagedAgent] = OrderedDict()
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> AlphaAgent:
        """Return the cached agent for a session, creating it when absent."""

        with self._lock:
            now = time.monotonic()
            self.evict_idle(now=now)
            entry = self._agents.get(session_id)
            if entry is None:
                entry = ManagedAgent(agent=self.factory.create(), last_used_at=now)
                self._agents[session_id] = entry
                self._evict_overflow()
            else:
                entry.last_used_at = now
                self._agents.move_to_end(session_id)
            return entry.agent

    def cancel(self, session_id: str) -> None:
        """Cancel an in-flight or next turn for a cached session."""

        with self._lock:
            entry = self._agents.get(session_id)
            if entry is not None:
                entry.agent.cancel(session_id)

    def evict_idle(self, *, now: float | None = None) -> None:
        """Evict agents idle longer than the configured TTL."""

        with self._lock:
            current = time.monotonic() if now is None else now
            expired = [
                session_id
                for session_id, entry in self._agents.items()
                if current - entry.last_used_at > self.idle_ttl_seconds
            ]
            for session_id in expired:
                self._release(session_id)

    def evict_all(self) -> None:
        """Release all cached agents."""

        with self._lock:
            for session_id in list(self._agents):
                self._release(session_id)

    def _evict_overflow(self) -> None:
        while len(self._agents) > self.max_size:
            session_id, _entry = next(iter(self._agents.items()))
            self._release(session_id)

    def _release(self, session_id: str) -> None:
        self._agents.pop(session_id, None)
