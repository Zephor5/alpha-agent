from __future__ import annotations

from pathlib import Path

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import Capability, CognitiveEventKind, ConfidenceCurve, SelfModel
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory


def test_subject_projection_current_includes_rebuilt_self_model(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    model = SelfModel(
        capabilities_self_assessed={
            Capability("summarize"): ConfidenceCurve("confidence=0.900;success=4;failure=0")
        }
    )
    event = emitter.emit(
        CognitiveEventKind.SELF_MODEL_UPDATED,
        payload={
            "before": SelfModel().to_record(),
            "after": model.to_record(),
            "subject": {"self_model": model.to_record()},
        },
    )
    projection = SubjectProjection(log, store)
    projection.apply(event)

    rebuilt = SubjectProjection(log, store)
    current = rebuilt.current()

    assert current.self_model.capabilities_self_assessed == {
        "summarize": "confidence=0.900;success=4;failure=0"
    }
