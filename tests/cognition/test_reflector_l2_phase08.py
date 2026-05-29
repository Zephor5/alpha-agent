from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
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
    Procedure,
    ProcedureId,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
    Step,
    Stimulus,
    StimulusKind,
    StrategyId,
    StrategyOverride,
    ThreadId,
    TriggerPattern,
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
from alpha_agent.cognition.reflectors.l2_rules.feedback_surprise_streak import (
    feedback_surprise_streak,
)
from alpha_agent.cognition.reflectors.l2_rules.lens_shift_flap import lens_shift_flap
from alpha_agent.cognition.stages.interpret import Interpreter
from alpha_agent.cognition.stages.types import AttentionFocus, Feedback
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.state.store import StateStore
from alpha_agent.tools.registry import ToolRegistry
from tests.cognition.helpers import clock_factory, id_factory


def test_reflector_l2_rules_emit_strategy_overrides(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    for index in range(3):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.REFLECTED,
            {
                "tick_id": f"tick-r-{index}",
                "reflections": [
                    _reflection(
                        f"reflection:{index}",
                        "contradiction-accepted",
                        f"2026-01-01T00:0{index}:00+00:00",
                    ).to_record()
                ],
            },
        )
    for _index in range(5):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.RECEIVED_FEEDBACK,
            {"matched_expected": False, "trigger": "summarize"},
        )
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.BELIEF_FORMED,
            {"origin": "novel_auto_form"},
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

    assert report.emitted == 4
    assert names == {
        "require_explicit_confirm_on_contradiction",
        "disable_auto_procedure_match_for_trigger",
        "freeze_lens_learning_for_24h",
        "require_confirm_before_novel_form",
    }


def test_feedback_surprise_streak_requires_consecutive_misses_for_same_trigger(
    tmp_path: Path,
) -> None:
    _store, log, projections, emitter = _runtime(tmp_path)
    for payload in [
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": True, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
    ]:
        _emit_apply(emitter, projections, CognitiveEventKind.RECEIVED_FEEDBACK, payload)

    interrupted = feedback_surprise_streak(
        [],
        list(log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK])),
        [],
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.RECEIVED_FEEDBACK,
        {"matched_expected": False, "trigger": "summarize"},
    )
    consecutive = feedback_surprise_streak(
        [],
        list(log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK])),
        [],
    )

    assert interrupted is None
    assert consecutive is not None
    assert consecutive["strategy_name"] == "disable_auto_procedure_match_for_trigger"


def test_feedback_surprise_streak_uses_current_streak_not_old_history(tmp_path: Path) -> None:
    _store, log, projections, emitter = _runtime(tmp_path)
    for payload in [
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": False, "trigger": "summarize"},
        {"matched_expected": True, "trigger": "summarize"},
    ]:
        _emit_apply(emitter, projections, CognitiveEventKind.RECEIVED_FEEDBACK, payload)

    assert feedback_surprise_streak(
        [],
        list(log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK])),
        [],
    ) is None


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
        "require_confirm_before_novel_form",
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


def test_strategy_application_in_interpret_revise_and_lens_learning(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    confirm_strategy = _strategy(
        "strategy:confirm",
        "require_confirm_before_novel_form",
        stages=["interpret"],
    )
    contradiction_strategy = _strategy(
        "strategy:contradiction",
        "require_explicit_confirm_on_contradiction",
        stages=["revise"],
    )
    freeze_strategy = _strategy(
        "strategy:freeze",
        "freeze_lens_learning_for_24h",
        stages=["lens_learning"],
    )
    for strategy in (confirm_strategy, contradiction_strategy, freeze_strategy):
        _emit_apply(
            emitter,
            projections,
            CognitiveEventKind.STRATEGY_CHANGED,
            {"strategy": strategy.to_record()},
        )

    interpretation = Interpreter().interpret(
        AttentionFocus(
            entities=[],
            salient_claims=[NLStatement("Novel safety claim.")],
            value_signals={},
        ),
        _empty_window(),
        [],
        projections.get_typed(SubjectProjection).current(),
        emitter=emitter,
        tick_id="tick-interpret",
        causal_parent=emitter.emit(CognitiveEventKind.ATTENDED, payload={}).id,
        strategies=[confirm_strategy],
    )
    assert interpretation.value.requires_confirmation is True

    controller = _controller(log, projections)
    contradictory = interpretation.value.__class__(
        stance="contradicting",
        supporting_beliefs=[],
        contradicting_beliefs=[],
        novel_claims=[],
        ambiguity_notes=[],
    )
    revised = controller.reviser.derive(
        Feedback(matched_expected=True),
        [],
        [],
        emitter=emitter,
        tick_id="tick-revise",
        causal_parent=interpretation.event.id,
        interpretation=contradictory,
        strategies=[contradiction_strategy],
    )
    assert revised.value == []
    pending = list(log.iter(kinds=[CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION]))
    assert pending[-1].payload["reason"] == (
        "strategy:require_explicit_confirm_on_contradiction"
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


def test_controller_skips_procedure_match_for_trigger_strategy(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    procedure = Procedure(
        id=ProcedureId("procedure:hello"),
        trigger=TriggerPattern("hello"),
        steps=[Step("respond with known path")],
        expected_outcome=NLStatement("ok"),
        learned_from=[],
        success_count=5,
        failure_count=0,
        confidence=0.9,
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.PROCEDURE_LEARNED,
        {"procedure": procedure.to_record(), "name": "Hello procedure"},
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {
            "strategy": _strategy(
                "strategy:disable-hello",
                "disable_auto_procedure_match_for_trigger",
                stages=["decide"],
                payload={"trigger": "hello"},
            ).to_record()
        },
    )

    result = CognitiveController(
        event_log=log,
        projections=projections,
        llm=_StaticProvider(),
        tools=ToolRegistry(),
        emitter=emitter,
    ).reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=None,
            payload="hello",
            thread_id=ThreadId.from_session("s1"),
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=ThreadId.from_session("s1"),
    )

    assert result.decision.action == "respond"
    decided = list(log.iter(kinds=[CognitiveEventKind.DECIDED]))[-1]
    assert decided.payload["procedure_count"] == 0


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
    if kind == CognitiveEventKind.RECEIVED_FEEDBACK and "tick_id" not in payload:
        payload = {**payload, "tick_id": f"tick-feedback-{emitter.log.length() + 1}"}
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


def _strategy(
    strategy_id: str,
    name: str,
    *,
    stages: list[str] | None = None,
    payload: dict[str, object] | None = None,
    for_counterpart=None,
    valid_until: str = "2027-01-01T00:00:00+00:00",
) -> StrategyOverride:
    return StrategyOverride(
        id=StrategyId(strategy_id),
        name=name,
        payload=payload or {},
        target_stages=stages or ["interpret", "decide", "revise", "lens_learning"],
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


def _lens_shift_payload(kind: str, before: float, after: float) -> dict[str, object]:
    return {
        "before": {"sensitivity": {kind: before}, "priorities": ["safety", "efficiency"]},
        "after": {"sensitivity": {kind: after}, "priorities": ["safety", "efficiency"]},
        "trigger": f"learn_value_lens observed 5 {kind} wins",
    }


def _empty_window():
    from alpha_agent.cognition.models import (
        ContextWindow,
        Instant,
        SituationId,
        Subject,
        situation_ref,
        subject_ref,
    )

    return ContextWindow(
        thread_id=ThreadId.from_session("test"),
        counterpart=None,
        foreground=[],
        background=None,
        recalled=[],
        recent_judgments=[],
        matched_procedures=[],
        subject_at=subject_ref(Subject().id),
        situation_at=situation_ref(SituationId("situation:test")),
        assembled_at=Instant("2026-01-01T00:00:00+00:00"),
    )


def _controller(log, projections):
    return CognitiveController(
        event_log=log,
        projections=projections,
        llm=_StaticProvider(),
        tools=ToolRegistry(),
    )


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


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
