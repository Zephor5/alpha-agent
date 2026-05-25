from datetime import timedelta

import pytest

from alpha_agent.cognition.coordinator import LockBusy, LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_try_acquire_busy_raises_without_blocking() -> None:
    coordinator = LoopCoordinator(SUBJECT_SELF)
    holder = LoopAcquireRequest("consolidation", LoopPriority.CONSOLIDATION, timedelta(seconds=30))
    reactive = LoopAcquireRequest("reactive", LoopPriority.REACTIVE, timedelta(seconds=1))

    with coordinator.acquire(holder):
        with pytest.raises(LockBusy) as exc_info:
            with coordinator.try_acquire(reactive):
                raise AssertionError("unreachable")

    assert exc_info.value.holder == "consolidation"
    assert exc_info.value.since
