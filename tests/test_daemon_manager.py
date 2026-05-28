from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alpha_agent.config import AlphaConfig, BashToolConfig
from alpha_agent.daemon.manager import AgentFactory, AgentManager
from alpha_agent.state.store import StateStore


class _FakeFactory:
    def __init__(self):
        self.created = 0
        self.agents: list[object] = []

    def create(self) -> object:
        self.created += 1
        agent = object()
        self.agents.append(agent)
        return agent


def _cached_session_ids(manager: AgentManager) -> set[str]:
    return set(manager._agents.keys())  # noqa: SLF001


def test_agent_manager_concurrent_get_or_create_returns_one_agent() -> None:
    factory = _FakeFactory()
    manager = AgentManager(factory)  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=8) as executor:
        agents = list(executor.map(lambda _: manager.get_or_create("s1"), range(32)))

    assert len({id(agent) for agent in agents}) == 1
    assert factory.created == 1


def test_agent_manager_evicts_idle_agents_before_reuse() -> None:
    factory = _FakeFactory()
    manager = AgentManager(
        factory,  # type: ignore[arg-type]
        idle_ttl_seconds=10,
    )

    first = manager.get_or_create("s1")
    manager.evict_idle(now=1_000_000_000)
    second = manager.get_or_create("s1")

    assert first is not second
    assert factory.created == 2


def test_agent_manager_keeps_agents_until_idle_ttl_is_exceeded(
    monkeypatch,
) -> None:
    monotonic = _MonotonicClock(100)
    monkeypatch.setattr("alpha_agent.daemon.manager.time.monotonic", monotonic)
    factory = _FakeFactory()
    manager = AgentManager(
        factory,  # type: ignore[arg-type]
        idle_ttl_seconds=10,
    )

    first = manager.get_or_create("s1")
    manager.evict_idle(now=110)
    at_boundary = manager.get_or_create("s1")
    manager.evict_idle(now=111)
    after_boundary = manager.get_or_create("s1")

    assert at_boundary is first
    assert after_boundary is not first
    assert factory.created == 2


def test_agent_manager_evicts_least_recently_used_agent_when_full() -> None:
    factory = _FakeFactory()
    manager = AgentManager(
        factory,  # type: ignore[arg-type]
        max_size=2,
    )

    first = manager.get_or_create("s1")
    second = manager.get_or_create("s2")
    assert manager.get_or_create("s1") is first

    third = manager.get_or_create("s3")

    assert manager.get_or_create("s1") is first
    assert manager.get_or_create("s2") not in {first, second, third}
    assert len(_cached_session_ids(manager)) == 2


def test_agent_factory_registers_configured_default_tools(tmp_path: Path) -> None:
    config = AlphaConfig(
        db_path=tmp_path / "alpha.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway-status.json",
        bash_tool=BashToolConfig(
            enabled=True,
            default_workdir=tmp_path,
            allowed_workdirs=(tmp_path,),
        ),
        tavily_api_key="tvly-test",
    )
    store = StateStore(config.db_path)
    store.initialize()

    agent = AgentFactory(config, store).create()

    assert agent.tool_registry.names() == ["bash", "web_search"]


class _MonotonicClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value
