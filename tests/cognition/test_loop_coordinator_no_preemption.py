from datetime import timedelta

import pytest

from alpha_agent.cognition.coordinator import LockBusy, LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_reactive_failures_do_not_preempt_holder() -> None:
    coordinator = LoopCoordinator(SUBJECT_SELF)
    holder = LoopAcquireRequest("consolidation", LoopPriority.CONSOLIDATION, timedelta(seconds=30))
    reactive = LoopAcquireRequest("reactive", LoopPriority.REACTIVE, timedelta(seconds=1))

    with coordinator.acquire(holder):
        for _ in range(5):
            with pytest.raises(LockBusy):
                with coordinator.try_acquire(reactive):
                    raise AssertionError("unreachable")
        assert coordinator.current_holder() == "consolidation"
