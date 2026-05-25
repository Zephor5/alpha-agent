from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops import (
    CheckpointStore,
    ConsolidationConfig,
    ConsolidationLoop,
    Scheduler,
)
from alpha_agent.cognition.loops.workers import (
    LearnValueLensWorker,
    ResolveQueuedConflictsWorker,
)
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CognitiveType,
    Reference,
    ValueKind,
    ValueProfile,
)
from alpha_agent.cognition.models.value import ValueLens
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.value import (
    default_value_lens,
    derive_value_profile,
    load_lens,
    resolve_conflict,
)
from alpha_agent.cognition.value.lens import save_lens
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_value_profile_derivation_combines_keywords_type_and_entities() -> None:
    profile = derive_value_profile(
        "Provide a safe accurate concise answer that helps the user learn.",
        cognitive_type=CognitiveType.PROCEDURAL,
        entities=[Reference("tool", "shell")],
    )

    assert profile.weights[ValueKind.SAFETY] > 0
    assert profile.weights[ValueKind.HONESTY] > 0
    assert profile.weights[ValueKind.EFFICIENCY] > 0
    assert profile.weights[ValueKind.LEARNING] > 0
    assert "type:procedural" in profile.notes
    assert "entity:tool" in profile.notes


def test_resolver_winner_and_tie_follow_lens() -> None:
    safety = _valued_belief("belief:safety", ValueKind.SAFETY)
    efficiency = _valued_belief("belief:efficiency", ValueKind.EFFICIENCY)

    default_resolution = resolve_conflict(safety, efficiency, default_value_lens())
    efficiency_first = resolve_conflict(
        safety,
        efficiency,
        ValueLens(priorities=[ValueKind.EFFICIENCY, ValueKind.SAFETY]),
    )
    tie = resolve_conflict(
        _valued_belief("belief:left", ValueKind.HONESTY),
        _valued_belief("belief:right", ValueKind.HONESTY),
        default_value_lens(),
    )

    assert default_resolution.winner_id == safety.id
    assert efficiency_first.winner_id == efficiency.id
    assert tie.tie is True


def test_queued_conflict_worker_supersedes_by_lens_and_is_idempotent(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    safety = _valued_belief("belief:safety", ValueKind.SAFETY)
    efficiency = _valued_belief("belief:efficiency", ValueKind.EFFICIENCY)
    for item in (safety, efficiency):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {"belief": item.to_record()},
        )
    conflict = _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED,
        {"belief_ids": [str(safety.id), str(efficiency.id)]},
    )

    reports = _run(log, projections, store, [ResolveQueuedConflictsWorker()])
    second_reports = _run(log, projections, store, [ResolveQueuedConflictsWorker()])

    assert reports[0].emitted == 1
    assert second_reports[0].emitted == 0
    assert projections.get_typed(BeliefProjection).get_by_id(efficiency.id).status == "superseded"
    event = list(log.iter(kinds=[CognitiveEventKind.BELIEF_SUPERSEDED]))[-1]
    assert event.payload["conflict_event_id"] == str(conflict.id)
    assert event.payload["decisive_value_kinds"] == ["safety", "efficiency"]
    assert "value_lens_explanation" in event.payload


def test_empty_profiles_are_derived_before_queued_conflict_resolution(tmp_path: Path) -> None:
    safety_winner = _empty_profile_conflict_winner(
        tmp_path / "safety",
        ValueLens(priorities=[ValueKind.SAFETY, ValueKind.EFFICIENCY]),
    )
    efficiency_winner = _empty_profile_conflict_winner(
        tmp_path / "efficiency",
        ValueLens(priorities=[ValueKind.EFFICIENCY, ValueKind.SAFETY]),
    )

    assert safety_winner == "belief:safety-empty"
    assert efficiency_winner == "belief:efficiency-empty"


def test_queued_conflict_worker_sends_tie_to_human_review(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    left = _valued_belief("belief:left", ValueKind.HONESTY)
    right = _valued_belief("belief:right", ValueKind.HONESTY)
    for item in (left, right):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {"belief": item.to_record()},
        )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED,
        {"belief_ids": [str(left.id), str(right.id)]},
    )

    reports = _run(log, projections, store, [ResolveQueuedConflictsWorker()])
    reviews = list(log.iter(kinds=[CognitiveEventKind.CONFLICT_KEPT_FOR_HUMAN_REVIEW]))

    assert reports[0].emitted == 1
    assert reviews[-1].payload["reason"] == "tie under current value lens"


def test_value_lens_learning_shifts_sensitivity_once_per_rate_window(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            {
                "old_belief_id": f"belief:old-{index}",
                "new_belief_id": f"belief:new-{index}",
                "decisive_value_kinds": ["efficiency"],
            },
        )

    first = _run(log, projections, store, [LearnValueLensWorker()])
    second = _run(log, projections, store, [LearnValueLensWorker()])
    shifted = list(log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED]))
    lens = load_lens(store)

    assert first[0].emitted == 1
    assert second[0].emitted == 0
    assert len(shifted) == 1
    assert lens.sensitivity[ValueKind.EFFICIENCY] == 1.1


def test_value_lens_learning_uses_event_order_for_rate_limit_now(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(SubjectProjection(log))
    projections.register(BeliefProjection(store))
    ids = iter(["shift", "z-old", "a-new-1", "a-new-2", "a-new-3", "a-new-4", "shift-2"])
    times = iter(
        [
            "2026-01-01T12:00:00+00:00",
            "2026-01-01T13:00:00+00:00",
            "2026-01-02T13:00:01+00:00",
            "2026-01-02T13:00:02+00:00",
            "2026-01-02T13:00:03+00:00",
            "2026-01-02T13:00:04+00:00",
            "2026-01-02T13:00:05+00:00",
        ]
    )
    emitter = EventEmitter(log, id_factory=lambda: next(ids), clock=lambda: next(times))
    save_lens(
        store,
        emitter,
        ValueLens(priorities=[ValueKind.EFFICIENCY, ValueKind.SAFETY]),
        trigger="baseline",
    )
    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            {
                "old_belief_id": f"belief:old-{index}",
                "new_belief_id": f"belief:new-{index}",
                "decisive_value_kinds": ["efficiency"],
            },
        )

    report = _run(log, projections, store, [LearnValueLensWorker()])[0]

    assert report.emitted == 1
    assert load_lens(store).sensitivity[ValueKind.EFFICIENCY] == 1.1


def test_value_lens_learning_counts_only_new_events_after_successful_checkpoint(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(SubjectProjection(log))
    projections.register(BeliefProjection(store))
    ids = iter(
        [
            "eff-1",
            "eff-2",
            "eff-3",
            "eff-4",
            "eff-5",
            "shift-eff",
            "safe-1",
            "safe-2",
            "safe-3",
            "safe-4",
            "safe-5",
            "shift-safe",
        ]
    )
    times = iter(
        [
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:02+00:00",
            "2026-01-01T00:00:03+00:00",
            "2026-01-01T00:00:04+00:00",
            "2026-01-01T00:00:05+00:00",
            "2026-01-01T00:00:06+00:00",
            "2026-01-02T00:00:07+00:00",
            "2026-01-02T00:00:08+00:00",
            "2026-01-02T00:00:09+00:00",
            "2026-01-02T00:00:10+00:00",
            "2026-01-02T00:00:11+00:00",
            "2026-01-02T00:00:12+00:00",
        ]
    )
    emitter = EventEmitter(log, id_factory=lambda: next(ids), clock=lambda: next(times))
    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            {
                "old_belief_id": f"belief:eff-old-{index}",
                "new_belief_id": f"belief:eff-new-{index}",
                "decisive_value_kinds": ["efficiency"],
            },
        )
    first = _run(log, projections, store, [LearnValueLensWorker()], emitter=emitter)[0]
    assert first.emitted == 1

    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            {
                "old_belief_id": f"belief:safe-old-{index}",
                "new_belief_id": f"belief:safe-new-{index}",
                "decisive_value_kinds": ["safety"],
            },
        )
    second = _run(log, projections, store, [LearnValueLensWorker()], emitter=emitter)[0]
    lens = load_lens(store)

    assert second.emitted == 1
    assert lens.sensitivity[ValueKind.EFFICIENCY] == 1.1
    assert lens.sensitivity[ValueKind.SAFETY] == 1.1


def test_cli_lens_show_and_set(tmp_path: Path) -> None:
    runner = CliRunner()
    env = _env(tmp_path)

    set_result = runner.invoke(
        app,
        ["cognition", "lens", "set", "--priority", "efficiency,safety,honesty"],
        env=env,
    )
    show_result = runner.invoke(app, ["cognition", "lens", "show"], env=env)

    assert set_result.exit_code == 0
    assert "value_lens_shifted" in set_result.output
    assert show_result.exit_code == 0
    assert "priority=efficiency,safety,honesty" in show_result.output


def test_subject_projection_rebuilds_value_lens_from_event_log(tmp_path: Path) -> None:
    store, log, _projections, emitter = _runtime(tmp_path)
    save_lens(
        store,
        emitter,
        ValueLens(priorities=[ValueKind.EFFICIENCY, ValueKind.SAFETY]),
        trigger="test",
    )
    with store.transaction() as conn:
        conn.execute("DELETE FROM subject_value_lens")

    projection = SubjectProjection(log)
    for event in log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED]):
        projection.apply(event)

    assert projection.current().value_lens.priorities[:2] == [
        ValueKind.EFFICIENCY,
        ValueKind.SAFETY,
    ]


def _runtime(tmp_path: Path):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(SubjectProjection(log))
    projections.register(BeliefProjection(store))
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())
    return store, log, projections, emitter


def _run(log, projections, store, workers, *, emitter=None):
    scheduler = Scheduler(log, CheckpointStore(store))
    return ConsolidationLoop(
        scheduler=scheduler,
        log=log,
        projections=projections,
        emitter=emitter,
        config=ConsolidationConfig(),
        workers=workers,
    ).run_once()


def _emit_apply(emitter, projections, kind, payload):
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


def _valued_belief(belief_id: str, value: ValueKind):
    return belief(
        belief_id,
        f"{value.value} claim.",
        object_="phase07-conflict",
    ).__class__.from_record(
        {
            **belief(
                belief_id,
                f"{value.value} claim.",
                object_="phase07-conflict",
            ).to_record(),
            "value_profile": ValueProfile(weights={value: 1.0}).to_record(),
        }
    )


def _empty_profile_conflict_winner(tmp_path: Path, lens: ValueLens) -> str:
    store, log, projections, emitter = _runtime(tmp_path)
    save_lens(store, emitter, lens, trigger="test")
    safety = belief(
        "belief:safety-empty",
        "A safe answer reduces harm.",
        object_="phase07-conflict",
    )
    efficiency = belief(
        "belief:efficiency-empty",
        "An efficient concise answer is faster.",
        object_="phase07-conflict",
    )
    for item in (safety, efficiency):
        assert not item.value_profile.weights
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {"belief": item.to_record()},
        )
    materialized = projections.get_typed(BeliefProjection)
    assert materialized.get_by_id(safety.id).value_profile.weights[ValueKind.SAFETY] > 0
    assert materialized.get_by_id(efficiency.id).value_profile.weights[ValueKind.EFFICIENCY] > 0
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED,
        {"belief_ids": [str(safety.id), str(efficiency.id)]},
    )

    _run(log, projections, store, [ResolveQueuedConflictsWorker()])

    event = list(log.iter(kinds=[CognitiveEventKind.BELIEF_SUPERSEDED]))[-1]
    return str(event.payload["new_belief_id"])


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }
