from datetime import timedelta
from threading import Event, Thread

from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_holder_yield_lets_higher_priority_waiter_run_then_resumes() -> None:
    coordinator = LoopCoordinator(SUBJECT_SELF)
    low = LoopAcquireRequest("consolidation", LoopPriority.CONSOLIDATION, timedelta(seconds=30))
    high = LoopAcquireRequest("drive", LoopPriority.DRIVE, timedelta(seconds=30))
    high_waiting = Event()
    high_done = Event()
    order: list[str] = []

    def run_high() -> None:
        high_waiting.set()
        with coordinator.acquire(high):
            order.append("high")
        high_done.set()

    with coordinator.acquire(low):
        order.append("low-before-yield")
        thread = Thread(target=run_high)
        thread.start()
        assert high_waiting.wait(1)
        while not coordinator.waiting():
            pass
        assert coordinator.yield_to_higher_priority() is True
        order.append("low-after-yield")
        assert coordinator.current_holder() == "consolidation"

    thread.join(timeout=1)
    assert high_done.is_set()
    assert order == ["low-before-yield", "high", "low-after-yield"]
