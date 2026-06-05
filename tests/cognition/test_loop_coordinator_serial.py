from datetime import timedelta
from threading import Event, Thread

from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_acquire_serializes_scheduled_loops() -> None:
    coordinator = LoopCoordinator(SUBJECT_SELF)
    first = LoopAcquireRequest("drive", LoopPriority.DRIVE, timedelta(seconds=30))
    second = LoopAcquireRequest(
        "consolidation",
        LoopPriority.CONSOLIDATION,
        timedelta(seconds=30),
    )
    acquired_second = Event()
    release_first = Event()
    order: list[str] = []

    def run_second() -> None:
        with coordinator.acquire(second):
            order.append("second")
            acquired_second.set()

    with coordinator.acquire(first):
        order.append("first")
        thread = Thread(target=run_second)
        thread.start()
        assert not acquired_second.wait(0.05)
        release_first.set()

    thread.join(timeout=1)
    assert acquired_second.is_set()
    assert order == ["first", "second"]
    assert release_first.is_set()
