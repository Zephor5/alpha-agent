from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

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
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
)
from alpha_agent.cognition.loops.workers import (
    ArchiveExpiredWorker,
    CompressContextWorker,
    LearnProcedureWorker,
    MergeBeliefsWorker,
    PromoteJudgmentWorker,
    SummarizeCounterpartWorker,
)
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartId,
    ExpectedFeedback,
    Instant,
    Procedure,
    ProcedureId,
    Step,
    Stimulus,
    StimulusKind,
    Subject,
    ThreadId,
    TriggerPattern,
    counterpart_ref,
)
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.stages.perceive import Perceiver
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, counterpart_payload, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_merge_beliefs_archives_expired_and_is_idempotent(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    belief_projection = projections.get_typed(BeliefProjection)
    first = belief("belief:a", "User prefers Python.", confidence=0.6)
    second = belief("belief:b", "User prefers Python.", confidence=0.9)
    expired = belief("belief:expired", "Temporary fact.").to_record()
    expired["applicability"] = '{"valid_until":"2020-01-01T00:00:00+00:00"}'
    for item in [first.to_record(), second.to_record(), expired]:
        _emit_apply(emitter, projections, CognitiveEventKind.BELIEF_FORMED, {"belief": item})

    reports = _run(
        log,
        projections,
        store,
        [MergeBeliefsWorker(), ArchiveExpiredWorker()],
    )
    second_reports = _run(
        log,
        projections,
        store,
        [MergeBeliefsWorker(), ArchiveExpiredWorker()],
    )

    assert belief_projection.get_by_id("belief:a").status == "superseded"
    assert belief_projection.get_by_id("belief:b").status == "active"
    assert belief_projection.get_by_id("belief:expired").status == "archived"
    assert sum(item.emitted for item in reports) == 2
    assert sum(item.emitted for item in second_reports) == 0


def test_promote_judgment_and_learn_procedure_are_idempotent(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    for index in range(3):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.JUDGED,
            {"tick_id": f"tick-j-{index}", "claim": "User prefers concise plans."},
        )
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.DECIDED,
            {
                "tick_id": f"tick-d-{index}",
                "action": "respond",
                "message": "summarize phase status",
            },
        )
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.RECEIVED_FEEDBACK,
            {"tick_id": f"tick-d-{index}", "matched_expected": True},
        )

    reports = _run(
        log,
        projections,
        store,
        [PromoteJudgmentWorker(), LearnProcedureWorker()],
    )
    second_reports = _run(
        log,
        projections,
        store,
        [PromoteJudgmentWorker(), LearnProcedureWorker()],
    )

    assert len(projections.get_typed(ProcedureProjection).list_active()) == 1
    active_beliefs = projections.get_typed(BeliefProjection).list_active()
    assert any(item.content == "User prefers concise plans." for item in active_beliefs)
    assert sum(item.emitted for item in reports) == 2
    assert sum(item.emitted for item in second_reports) == 0


def test_compress_context_moves_old_foreground_into_background(tmp_path) -> None:
    store, log, projections, _emitter = _runtime(tmp_path, context_recent_limit=20)
    context = projections.get_typed(ContextWindowProjection)
    thread_id = ThreadId.from_session("s1")
    perception_ids = []
    for index in range(10):
        event = Perceiver().perceive(
            Stimulus(
                kind=StimulusKind.USER_MESSAGE,
                source=None,
                payload=f"message-{index}",
                thread_id=thread_id,
                received_at=Instant("2026-01-01T00:00:00+00:00"),
            ),
            Subject(),
            emitter=EventEmitter(log),
            tick_id=f"tick-{index}",
        ).event
        context.apply(event)
        perception_ids.append(event.payload["perception"]["id"])

    report = _run(
        log,
        projections,
        store,
        [CompressContextWorker()],
        config=ConsolidationConfig(context_foreground_max=6, context_absorb_batch=4),
    )[0]
    window = context.get(thread_id, Subject())

    assert report.emitted == 1
    assert [item.raw for item in window.foreground] == [
        f"message-{index}" for index in range(4, 10)
    ]
    assert window.background is not None
    event = list(log.iter(kinds=[CognitiveEventKind.CONTEXT_COMPRESSED]))[-1]
    assert event.payload["absorbed_perception_ids"] == perception_ids[:4]
    with store.connect() as conn:
        row = conn.execute(
            "SELECT summary, derived_from_perception_ids FROM context_window_background"
        ).fetchone()
    assert "message-0" in row["summary"]
    assert perception_ids[0] in row["derived_from_perception_ids"]


def test_counterpart_digest_supersedes_and_replays(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    counterpart_id = "counterpart:user-a"
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
        counterpart_payload(counterpart_id),
    )
    counterpart = counterpart_ref(CounterpartId(counterpart_id))
    for index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {
                "belief": belief(
                    f"belief:user-a-{index}",
                    f"User A preference {index}.",
                    about=[counterpart],
                    object_=f"pref-{index}",
                ).to_record()
            },
        )

    first = _run(log, projections, store, [SummarizeCounterpartWorker()])[0]
    first_digest = _active_digest_ids(projections.get_typed(BeliefProjection), counterpart_id)
    for index in range(5, 8):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {
                "belief": belief(
                    f"belief:user-a-{index}",
                    f"User A preference {index}.",
                    about=[counterpart],
                    object_=f"pref-{index}",
                ).to_record()
            },
        )
    second = _run(log, projections, store, [SummarizeCounterpartWorker()])[0]
    second_digest = _active_digest_ids(projections.get_typed(BeliefProjection), counterpart_id)

    assert first.emitted == 1
    assert second.emitted == 2
    assert first_digest != second_digest
    assert len(second_digest) == 1

    with store.immediate_transaction() as conn:
        conn.execute("DROP TABLE belief_view")
        conn.execute("DROP TABLE belief_about_index")
        conn.execute("DROP TABLE belief_entity_index")
    rebuilt = BeliefProjection(store)
    registry = ProjectionRegistry()
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()
    assert _active_digest_ids(rebuilt, counterpart_id) == second_digest


def test_checkpoint_persistence_and_cli_dry_run(tmp_path, monkeypatch) -> None:
    store, _log, _projections, _emitter = _runtime(tmp_path)
    checkpoints = CheckpointStore(store)
    checkpoints.save(
        WorkerCheckpoint(
            worker_name="merge_beliefs",
            last_run_at=Instant("2026-01-01T00:00:00+00:00"),
            last_processed_event_id=None,
            last_status="ok",
            metadata={"cursor": "belief:a"},
        )
    )
    assert checkpoints.load("merge_beliefs").metadata == {"cursor": "belief:a"}

    db_path = tmp_path / "cli.db"
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    runner = CliRunner()
    result = runner.invoke(app, ["cognition", "consolidate", "--now", "--dry-run"])

    assert result.exit_code == 0
    assert "dry_run=true" in result.output


def test_cli_dry_run_does_not_mutate_real_db(tmp_path, monkeypatch) -> None:
    store, _log, projections, emitter = _runtime(tmp_path)
    first = belief("belief:a", "User prefers Python.", confidence=0.6)
    second = belief("belief:b", "User prefers Python.", confidence=0.9)
    for item in [first, second]:
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {"belief": item.to_record()},
        )
    before = _table_counts(
        store,
        [
            "cognitive_events",
            "belief_view",
            "context_window_view",
            "context_window_background",
            "procedure_view",
            "cognition_worker_checkpoint",
        ],
    )

    monkeypatch.setenv("ALPHA_DB_PATH", str(store.db_path))
    result = CliRunner().invoke(
        app,
        ["cognition", "consolidate", "--now", "--dry-run"],
    )

    assert result.exit_code == 0
    assert _table_counts(store, list(before)) == before
    assert projections.get_typed(BeliefProjection).get_by_id("belief:a").status == "active"


def test_scheduler_tick_gates_workers_and_updates_backlog_cursor(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    worker = _CountingWorker()
    scheduler = Scheduler(log, CheckpointStore(store))
    scheduler.register(worker)
    coordinator = _CountingCoordinator()
    config = ConsolidationConfig()

    reports = scheduler.tick(
        Instant("2026-01-01T00:00:00+00:00"),
        coordinator=coordinator,
        projections=projections,
        emitter=emitter,
        config=config,
    )
    assert reports == []
    assert coordinator.acquire_count == 0

    first = _emit_apply(emitter, projections, CognitiveEventKind.JUDGED, {"claim": "one"})
    reports = scheduler.tick(
        Instant("2026-01-01T00:01:00+00:00"),
        coordinator=coordinator,
        projections=projections,
        emitter=emitter,
        config=config,
    )
    assert reports == []
    assert coordinator.acquire_count == 0

    second = _emit_apply(emitter, projections, CognitiveEventKind.JUDGED, {"claim": "two"})
    reports = scheduler.tick(
        Instant("2026-01-01T00:02:00+00:00"),
        coordinator=coordinator,
        projections=projections,
        emitter=emitter,
        config=config,
    )
    assert [item.worker for item in reports] == ["counting"]
    assert coordinator.acquire_count == 1
    checkpoint = CheckpointStore(store).load("counting")
    assert checkpoint.last_processed_event_id == second.id

    _emit_apply(emitter, projections, CognitiveEventKind.JUDGED, {"claim": "three"})
    reports = scheduler.tick(
        Instant("2026-01-01T00:03:00+00:00"),
        coordinator=coordinator,
        projections=projections,
        emitter=emitter,
        config=config,
    )
    assert reports == []
    assert coordinator.acquire_count == 1
    assert checkpoint.last_processed_event_id != first.id


def test_scheduler_resumes_yielded_checkpoint_without_min_interval_wait(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    worker = _SlowIntervalWorker()
    scheduler = Scheduler(log, CheckpointStore(store))
    scheduler.register(worker)
    CheckpointStore(store).save(
        WorkerCheckpoint(
            worker_name=worker.name,
            last_run_at=Instant("2026-01-01T00:00:00+00:00"),
            last_status="yielded",
            metadata={"last_claim": "one"},
        )
    )

    reports = scheduler.tick(
        Instant("2026-01-01T00:01:00+00:00"),
        coordinator=_CountingCoordinator(),
        projections=projections,
        emitter=emitter,
        config=ConsolidationConfig(),
    )

    assert [item.worker for item in reports] == [worker.name]


def test_worker_resume_wraps_to_lower_items_before_backlog_cursor_advances(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    _seed_counterpart_beliefs(emitter, projections, "counterpart:a", 5)
    _seed_counterpart_beliefs(emitter, projections, "counterpart:c", 5)
    CheckpointStore(store).save(
        WorkerCheckpoint(
            worker_name="summarize_counterpart",
            last_status="yielded",
            metadata={"last_counterpart_id": "counterpart:b"},
        )
    )

    reports = _run(log, projections, store, [SummarizeCounterpartWorker()])
    checkpoint = CheckpointStore(store).load("summarize_counterpart")

    assert reports[0].emitted == 2
    assert _active_digest_ids(projections.get_typed(BeliefProjection), "counterpart:a")
    assert _active_digest_ids(projections.get_typed(BeliefProjection), "counterpart:c")
    assert checkpoint.last_processed_event_id is not None


def test_noop_worker_scan_yields_at_chunk_boundary(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.BELIEF_FORMED,
        {"belief": belief("belief:not-expired", "Still active.").to_record()},
    )

    reports = _run(
        log,
        projections,
        store,
        [ArchiveExpiredWorker()],
        coordinator=_YieldOnceCoordinator(),
    )
    checkpoint = CheckpointStore(store).load("archive_expired")

    assert reports[0].yielded_to_higher_priority is True
    assert reports[0].emitted == 0
    assert checkpoint.metadata == {"last_belief_id": "belief:not-expired"}


def test_summarize_counterpart_resumes_after_checkpoint_cursor(tmp_path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    _seed_counterpart_beliefs(emitter, projections, "counterpart:a", 5)
    _seed_counterpart_beliefs(emitter, projections, "counterpart:b", 5)

    first_reports = _run(
        log,
        projections,
        store,
        [SummarizeCounterpartWorker()],
        coordinator=_YieldOnceCoordinator(),
    )
    checkpoint = CheckpointStore(store).load("summarize_counterpart")
    assert first_reports[0].yielded_to_higher_priority is True
    assert checkpoint.metadata == {"last_counterpart_id": "counterpart:a"}
    assert _active_digest_ids(projections.get_typed(BeliefProjection), "counterpart:a")
    assert not _active_digest_ids(projections.get_typed(BeliefProjection), "counterpart:b")

    second_reports = _run(log, projections, store, [SummarizeCounterpartWorker()])
    checkpoint = CheckpointStore(store).load("summarize_counterpart")
    assert second_reports[0].emitted == 1
    assert checkpoint.metadata == {}
    assert _active_digest_ids(projections.get_typed(BeliefProjection), "counterpart:b")


def test_procedure_projection_rebuilds_from_event_log(tmp_path) -> None:
    store, log, _projections, emitter = _runtime(tmp_path)
    procedure = Procedure(
        id=ProcedureId("procedure:test"),
        trigger=TriggerPattern("respond:summarize"),
        steps=[Step("repeat action pattern: respond:summarize")],
        expected_outcome=ExpectedFeedback("matched_expected_feedback"),
        learned_from=[],
        success_count=3,
        failure_count=0,
        confidence=0.8,
    )
    emitter.emit(
        CognitiveEventKind.PROCEDURE_LEARNED,
        payload={"procedure": procedure.to_record(), "name": "Repeat summarize"},
    )
    with store.immediate_transaction() as conn:
        conn.execute("DROP TABLE procedure_view")

    rebuilt = ProcedureProjection(store, event_log=log, auto_rebuild=True)

    assert rebuilt.get("procedure:test") == procedure


def _runtime(tmp_path, *, context_recent_limit: int = 12):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(BeliefProjection(store))
    projections.register(CounterpartProjection(store))
    projections.register(ProcedureProjection(store))
    projections.register(ContextWindowProjection(log, recent_limit=context_recent_limit))
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())
    return store, log, projections, emitter


def _run(
    log,
    projections,
    store,
    workers,
    *,
    config: ConsolidationConfig | None = None,
    coordinator=None,
):
    scheduler = Scheduler(log, CheckpointStore(store))
    return ConsolidationLoop(
        scheduler=scheduler,
        coordinator=coordinator,
        log=log,
        projections=projections,
        config=config or ConsolidationConfig(),
        workers=workers,
    ).run_once()


def _emit_apply(emitter, projections, kind, payload):
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


def _active_digest_ids(projection: BeliefProjection, counterpart_id: str) -> list[str]:
    return [
        str(item.id)
        for item in projection.recall_about(counterpart_ref(CounterpartId(counterpart_id)))
        if item.object == f"counterpart_digest:{counterpart_id}"
    ]


def _seed_counterpart_beliefs(emitter, projections, counterpart_id: str, count: int) -> None:
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
        counterpart_payload(counterpart_id),
    )
    counterpart = counterpart_ref(CounterpartId(counterpart_id))
    for index in range(count):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {
                "belief": belief(
                    f"belief:{counterpart_id}:{index}",
                    f"{counterpart_id} preference {index}.",
                    about=[counterpart],
                    object_=f"{counterpart_id}:pref-{index}",
                ).to_record()
            },
        )


def _table_counts(store: StateStore, tables: list[str]) -> dict[str, int]:
    with store.connect() as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])
            for table in tables
        }


class _CountingWorker:
    name = "counting"
    trigger = ScheduleTrigger(
        min_interval=timedelta(seconds=0),
        max_interval=None,
        watches=frozenset({CognitiveEventKind.JUDGED}),
        min_new_events=2,
    )
    handles_event_kinds = frozenset({CognitiveEventKind.JUDGED})

    def run(
        self,
        log,
        projections,
        emitter,
        coordinator,
        config,
        checkpoint,
    ) -> WorkerReport:
        del log, projections, emitter, coordinator, config
        return WorkerReport(
            worker=self.name,
            inspected=1,
            emitted=0,
            notes=[],
            yielded_to_higher_priority=False,
            new_checkpoint=checkpoint,
        )


class _SlowIntervalWorker(_CountingWorker):
    name = "slow"
    trigger = ScheduleTrigger(
        min_interval=timedelta(hours=6),
        max_interval=None,
        watches=frozenset({CognitiveEventKind.JUDGED}),
        min_new_events=99,
    )


class _CountingCoordinator:
    def __init__(self) -> None:
        self.acquire_count = 0

    @contextmanager
    def acquire(self, _req: LoopAcquireRequest) -> Iterator[None]:
        self.acquire_count += 1
        yield

    def yield_to_higher_priority(self) -> bool:
        return False


class _YieldOnceCoordinator(_CountingCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self._yielded = False

    def yield_to_higher_priority(self) -> bool:
        if self._yielded:
            return False
        self._yielded = True
        return True
