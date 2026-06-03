from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops import (
    CheckpointStore,
    ConsolidationConfig,
    ConsolidationLoop,
    Scheduler,
)
from alpha_agent.cognition.loops.workers import ExpireStrategiesWorker, LearnValueLensWorker
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    NLStatement,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
    StrategyId,
    StrategyOverride,
    counterpart_ref,
)
from alpha_agent.cognition.models._ids import CounterpartId
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import (
    StrategyProjection,
    strategy_applies_to_counterpart,
)
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.reflectors.l2 import ReflectorL2
from alpha_agent.cognition.reflectors.l2_rules.lens_shift_flap import lens_shift_flap
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory


def test_reflector_l2_rules_emit_strategy_overrides(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    for index in range(3):
        reflection = _reflection(
            f"reflection:{index}",
            "contradiction-accepted",
            f"2026-01-01T00:0{index}:00+00:00",
        )
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.REFLECTED,
            _reflected_payload(f"turn-r-{index}", [reflection]),
        )
    for index in range(3):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.VALUE_LENS_SHIFTED,
            _lens_shift_payload("efficiency", 1.0 + (index * 0.1), 1.1 + (index * 0.1)),
        )

    report = ReflectorL2().run(
        log,
        projections,
        emitter,
        _NoYieldCoordinator(),
        ConsolidationConfig(),
        CheckpointStore(store).load("reflector_l2"),
    )
    names = {strategy.name for strategy in projections.get_typed(StrategyProjection).active()}

    assert report.emitted == 2
    assert names == {
        "require_explicit_confirm_on_contradiction",
        "freeze_lens_learning_for_24h",
    }


def test_lens_shift_flap_groups_by_same_shift_direction(tmp_path: Path) -> None:
    _store, log, projections, emitter = _runtime(tmp_path)
    for payload in [
        _lens_shift_payload("efficiency", 1.0, 1.1),
        _lens_shift_payload("safety", 1.0, 1.1),
        _lens_shift_payload("efficiency", 1.1, 1.0),
    ]:
        _emit_apply(emitter, projections, CognitiveEventKind.VALUE_LENS_SHIFTED, payload)

    mixed = lens_shift_flap(
        [],
        list(log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED])),
        [],
    )
    for payload in [
        _lens_shift_payload("efficiency", 1.1, 1.2),
        _lens_shift_payload("efficiency", 1.2, 1.3),
        _lens_shift_payload("efficiency", 1.3, 1.4),
    ]:
        _emit_apply(emitter, projections, CognitiveEventKind.VALUE_LENS_SHIFTED, payload)
    same_direction = lens_shift_flap(
        [],
        list(log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED])),
        [],
    )

    assert mixed is None
    assert same_direction is not None
    assert same_direction["payload"]["direction"] == "sensitivity:efficiency:up"


def test_strategy_projection_caps_active_strategies_at_five(tmp_path: Path) -> None:
    _store, _log, projections, emitter = _runtime(tmp_path)
    projection = projections.get_typed(StrategyProjection)

    for index in range(6):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.STRATEGY_CHANGED,
            {"strategy": _strategy(f"strategy:{index}", f"phase08-test-{index}").to_record()},
        )

    active = projection.active(now="2026-01-01T00:30:00+00:00")

    assert len(active) == 5
    assert [str(item.id) for item in active] == [
        "strategy:0",
        "strategy:1",
        "strategy:2",
        "strategy:3",
        "strategy:4",
    ]


def test_strategy_projection_cap_blocks_reactivating_existing_expired_id(
    tmp_path: Path,
) -> None:
    _store, _log, projections, emitter = _runtime(tmp_path)
    projection = projections.get_typed(StrategyProjection)
    expired = _strategy(
        "strategy:expired-extra",
        "expired-extra",
        valid_until="2026-01-01T00:10:00+00:00",
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {"strategy": expired.to_record()},
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_EXPIRED,
        {"strategy_id": str(expired.id), "reason": "test"},
    )
    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.STRATEGY_CHANGED,
            {"strategy": _strategy(f"strategy:active-{index}", f"active-{index}").to_record()},
        )

    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {
            "strategy": _strategy(
                "strategy:expired-extra",
                "reactivated-extra",
                valid_until="2027-01-01T00:00:00+00:00",
            ).to_record()
        },
    )

    assert len(projection.active(now="2026-01-01T00:30:00+00:00")) == 5
    assert projection.active(now="2026-01-01T00:30:00+00:00")[-1].id == "strategy:active-4"


def test_strategy_projection_expiry_counterpart_matching_and_cli(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    global_strategy = _strategy(
        "strategy:global",
        "freeze_lens_learning_for_24h",
        valid_until="2026-01-01T01:00:00+00:00",
    )
    user_a = counterpart_ref(CounterpartId("counterpart:user-a"))
    user_b = counterpart_ref(CounterpartId("counterpart:user-b"))
    scoped = _strategy(
        "strategy:scoped",
        "require_explicit_confirm_on_contradiction",
        for_counterpart=user_a,
        valid_until="2026-01-01T01:00:00+00:00",
    )
    for strategy in (global_strategy, scoped):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.STRATEGY_CHANGED,
            {"strategy": strategy.to_record()},
        )

    projection = projections.get_typed(StrategyProjection)
    assert len(projection.active(now="2026-01-01T00:30:00+00:00")) == 2
    assert strategy_applies_to_counterpart(global_strategy, user_b)
    assert strategy_applies_to_counterpart(scoped, user_a)
    assert not strategy_applies_to_counterpart(scoped, user_b)

    reports = _run(
        log,
        projections,
        store,
        [ExpireStrategiesWorker()],
        config=ConsolidationConfig(),
        emitter=EventEmitter(
            log,
            id_factory=id_factory("expire"),
            clock=lambda: "2026-01-01T02:00:00+00:00",
        ),
    )
    assert reports[0].emitted == 2
    assert projection.active(now="2026-01-01T02:00:00+00:00") == []

    cli_store = StateStore(tmp_path / "cli.db")
    cli_store.initialize()
    cli_log = SQLiteEventLog(cli_store)
    cli_projection = StrategyProjection(cli_store)
    cli_emitter = EventEmitter(cli_log, id_factory=id_factory("cli"), clock=clock_factory())
    event = cli_emitter.emit(
        CognitiveEventKind.STRATEGY_CHANGED,
        payload={
            "strategy": _strategy(
                "strategy:cli",
                "freeze_lens_learning_for_24h",
            ).to_record()
        },
    )
    cli_projection.apply(event)
    env = _env(tmp_path, db_name="cli.db")
    list_result = CliRunner().invoke(app, ["cognition", "strategies", "--active"], env=env)
    expire_result = CliRunner().invoke(
        app,
        ["cognition", "strategy-expire", "strategy:cli"],
        env=env,
    )

    assert list_result.exit_code == 0
    assert "strategy:cli" in list_result.output
    assert expire_result.exit_code == 0
    assert "strategy_expired" in expire_result.output


def test_lens_learning_strategy_freeze_blocks_value_lens_worker(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    freeze_strategy = _strategy(
        "strategy:freeze",
        "freeze_lens_learning_for_24h",
        domains=["lens_learning"],
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {"strategy": freeze_strategy.to_record()},
    )

    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            {"decisive_value_kinds": ["safety"], "old_belief_id": index, "new_belief_id": index},
        )
    report = _run(log, projections, store, [LearnValueLensWorker()])[0]
    assert report.emitted == 0
    assert "lens learning frozen" in report.notes[0]


def _runtime(tmp_path: Path):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(SubjectProjection(log))
    projections.register(BeliefProjection(store))
    projections.register(ProcedureProjection(store))
    projections.register(ReflectionProjection(store))
    projections.register(StrategyProjection(store))
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    return store, log, projections, emitter


def _run(log, projections, store, workers, *, config=None, emitter=None):
    scheduler = Scheduler(log, CheckpointStore(store))
    return ConsolidationLoop(
        scheduler=scheduler,
        log=log,
        projections=projections,
        emitter=emitter,
        config=config or ConsolidationConfig(),
        workers=workers,
    ).run_once()


def _emit_apply(emitter, projections, kind, payload):
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


def _strategy(
    strategy_id: str,
    name: str,
    *,
    domains: list[str] | None = None,
    payload: dict[str, object] | None = None,
    for_counterpart=None,
    valid_until: str = "2027-01-01T00:00:00+00:00",
) -> StrategyOverride:
    return StrategyOverride(
        id=StrategyId(strategy_id),
        name=name,
        payload=payload or {},
        target_domains=domains or ["memory_propose", "lens_learning"],
        for_counterpart=for_counterpart,
        set_by="test",
        set_at=Instant("2026-01-01T00:00:00+00:00"),
        valid_until=Instant(valid_until),
    )


def _reflection(reflection_id: str, kind: str, created_at: str) -> Reflection:
    return Reflection(
        id=ReflectionId(reflection_id),
        level="L1",
        kind=ReflectionKind(kind),
        severity=Severity("warning"),
        target=ReflectionTarget("belief:test"),
        finding=NLStatement("finding"),
        suggested_remedy=RemedyHint(""),
        created_at=Instant(created_at),
    )


def _reflected_payload(turn_id: str, reflections: list[Reflection]) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "session_id": "s1",
        "reflection_count": len(reflections),
        "reflection_ids": [str(reflection.id) for reflection in reflections],
        "targets": [{"kind": "reflection", "id": str(reflection.id)} for reflection in reflections],
        "reflections": [reflection.to_record() for reflection in reflections],
    }


def _lens_shift_payload(kind: str, before: float, after: float) -> dict[str, object]:
    return {
        "before": {"sensitivity": {kind: before}, "priorities": ["safety", "efficiency"]},
        "after": {"sensitivity": {kind: after}, "priorities": ["safety", "efficiency"]},
        "trigger": f"learn_value_lens observed 5 {kind} wins",
    }


def _env(tmp_path: Path, *, db_name: str = "alpha.db") -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / db_name),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


class _NoYieldCoordinator:
    @contextmanager
    def acquire(self, _req: LoopAcquireRequest) -> Iterator[None]:
        yield

    def yield_to_higher_priority(self) -> bool:
        return False
