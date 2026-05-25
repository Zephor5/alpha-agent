from datetime import timedelta

from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_try_acquire_free_lock_enters_immediately() -> None:
    coordinator = LoopCoordinator(SUBJECT_SELF)
    req = LoopAcquireRequest("reactive", LoopPriority.REACTIVE, timedelta(seconds=1))

    with coordinator.try_acquire(req):
        assert coordinator.current_holder() == "reactive"

    assert coordinator.current_holder() is None
