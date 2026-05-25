"""Single-subject cooperative loop coordinator."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

from alpha_agent.cognition.models import Instant, SubjectId
from alpha_agent.cognition.models.enums import LoopPriority
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class LoopAcquireRequest:
    """Request for a loop to enter the single-subject critical section."""

    loop_name: str
    priority: LoopPriority
    max_chunk_duration: timedelta


class LockBusy(Exception):
    """Raised when non-blocking acquisition finds the subject lock busy."""

    def __init__(self, holder: str, since: Instant):
        super().__init__(f"cognition loop lock is held by {holder} since {since}")
        self.holder = holder
        self.since = since


@dataclass
class _Waiter:
    loop_name: str
    priority: LoopPriority
    sequence: int


@dataclass
class _YieldingHolder:
    loop_name: str
    priority: LoopPriority


class LoopCoordinator:
    """Cooperative lock for all loops running under one Subject."""

    def __init__(self, subject_id: SubjectId):
        self.subject_id = subject_id
        self._condition = threading.Condition()
        self._holder: str | None = None
        self._holder_priority: LoopPriority | None = None
        self._holder_since: Instant | None = None
        self._waiters: list[_Waiter] = []
        self._sequence = 0
        self._yielding_holder: _YieldingHolder | None = None

    @contextmanager
    def acquire(self, req: LoopAcquireRequest) -> Iterator[None]:
        """Block until a scheduled loop acquires the lock."""

        waiter = self._enqueue(req)
        with self._condition:
            self._condition.wait_for(lambda: self._can_acquire(waiter))
            self._waiters.remove(waiter)
            self._set_holder(req)
        try:
            yield
        finally:
            self._release_if_holder(req.loop_name)

    @contextmanager
    def try_acquire(self, req: LoopAcquireRequest) -> Iterator[None]:
        """Acquire immediately or raise LockBusy without entering the queue."""

        with self._condition:
            if self._holder is not None:
                raise LockBusy(self._holder, self._holder_since or Instant(""))
            self._set_holder(req)
        try:
            yield
        finally:
            self._release_if_holder(req.loop_name)

    def yield_to_higher_priority(self) -> bool:
        """Cooperatively release and reacquire around higher-priority waiters."""

        with self._condition:
            if self._holder is None or self._holder_priority is None:
                return False
            holder = _YieldingHolder(self._holder, self._holder_priority)
            has_higher_waiter = any(waiter.priority < holder.priority for waiter in self._waiters)
            self._yielding_holder = holder
            self._holder = None
            self._holder_priority = None
            self._holder_since = None
            self._condition.notify_all()
            if has_higher_waiter:
                self._condition.wait_for(
                    lambda: self._holder is None
                    and not any(waiter.priority < holder.priority for waiter in self._waiters)
                )
            else:
                self._condition.wait(timeout=0.01)
                self._condition.wait_for(lambda: self._holder is None)
            self._holder = holder.loop_name
            self._holder_priority = holder.priority
            self._holder_since = Instant(utc_now_iso())
            self._yielding_holder = None
            self._condition.notify_all()
            return has_higher_waiter

    def current_holder(self) -> str | None:
        with self._condition:
            return self._holder

    def waiting(self) -> list[tuple[str, LoopPriority]]:
        with self._condition:
            ordered = sorted(self._waiters, key=lambda waiter: (waiter.priority, waiter.sequence))
            return [(waiter.loop_name, waiter.priority) for waiter in ordered]

    def _enqueue(self, req: LoopAcquireRequest) -> _Waiter:
        with self._condition:
            self._sequence += 1
            waiter = _Waiter(req.loop_name, req.priority, self._sequence)
            self._waiters.append(waiter)
            self._condition.notify_all()
            return waiter

    def _can_acquire(self, waiter: _Waiter) -> bool:
        if self._holder is not None:
            return False
        if self._yielding_holder is not None and waiter.priority >= self._yielding_holder.priority:
            return False
        eligible = self._eligible_waiters()
        if not eligible:
            return False
        return waiter == min(eligible, key=lambda item: (item.priority, item.sequence))

    def _eligible_waiters(self) -> list[_Waiter]:
        if self._yielding_holder is None:
            return list(self._waiters)
        return [
            waiter
            for waiter in self._waiters
            if waiter.priority < self._yielding_holder.priority
        ]

    def _set_holder(self, req: LoopAcquireRequest) -> None:
        self._holder = req.loop_name
        self._holder_priority = req.priority
        self._holder_since = Instant(utc_now_iso())
        self._condition.notify_all()

    def _release_if_holder(self, loop_name: str) -> None:
        with self._condition:
            if self._holder == loop_name:
                self._holder = None
                self._holder_priority = None
                self._holder_since = None
                self._condition.notify_all()
