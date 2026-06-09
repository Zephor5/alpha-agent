"""Direct asynchronous extraction for freshly compacted handovers."""

from __future__ import annotations

import time
from collections.abc import Sequence
from threading import BoundedSemaphore, Lock, Thread, current_thread

from alpha_agent.cognition.loops.scheduler import WorkerReport
from alpha_agent.cognition.loops.workers.memory_extraction import MemoryExtractionWorker
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.llm.base import LLMProvider, LLMToolDefinitionInput
from alpha_agent.llm.tracing import LLMTraceLogger
from alpha_agent.runtime.context_handover import HandoverExtractionJob
from alpha_agent.state.store import StateStore


class DirectCompactExtractionService:
    """Submit compact extraction work immediately without using runtime traces as a queue."""

    def __init__(
        self,
        *,
        store: StateStore,
        llm_provider: LLMProvider,
        tools: Sequence[LLMToolDefinitionInput] = (),
        source_batch_size: int = 12,
        max_workers: int = 2,
        enabled: bool = True,
        llm_trace_logger: LLMTraceLogger | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.tools = tuple(tools)
        self.source_batch_size = max(1, int(source_batch_size))
        self.enabled = enabled
        self.llm_trace_logger = llm_trace_logger
        self._slots = BoundedSemaphore(max(1, int(max_workers)))
        self._lock = Lock()
        self._threads: set[Thread] = set()
        self._closed = False

    def submit(
        self,
        job: HandoverExtractionJob,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
    ) -> bool:
        """Start direct compact extraction in a daemon thread when capacity is available."""

        if not self.enabled:
            return False
        with self._lock:
            if self._closed:
                return False
        if not self._slots.acquire(blocking=False):
            self._write_audit(
                "direct_compact_extraction_saturated",
                {
                    "session_id": job.session_id,
                    "compressed_message_id": job.compressed_message_id,
                },
            )
            return False

        thread = Thread(
            target=self._run_job,
            args=(job, tuple(tools if tools is not None else self.tools)),
            name="alpha-direct-compact-extraction",
            daemon=True,
        )
        with self._lock:
            if self._closed:
                self._slots.release()
                return False
            self._threads.add(thread)
            thread.start()
        return True

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Prevent new submissions and optionally wait for already-started jobs."""

        with self._lock:
            self._closed = True
            threads = list(self._threads)
        if not wait:
            return
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        for thread in threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

    def _run_job(
        self,
        job: HandoverExtractionJob,
        tools: Sequence[LLMToolDefinitionInput],
    ) -> None:
        reports: list[WorkerReport] = []
        try:
            worker = MemoryExtractionWorker(
                CognitionStateStore(self.store),
                self.llm_provider,
                tools=tools,
                source_batch_size=self.source_batch_size,
                llm_trace_logger=self.llm_trace_logger,
            )
            while True:
                report = worker.run_compact_job(job)
                reports.append(report)
                if report.new_checkpoint.last_status != "ok" or report.emitted == 0:
                    break
            self._write_audit(
                "direct_compact_extraction_completed",
                {
                    "session_id": job.session_id,
                    "compressed_message_id": job.compressed_message_id,
                    "reports": [_report_record(report) for report in reports],
                },
            )
        except Exception as exc:
            self._write_audit(
                "direct_compact_extraction_failed",
                {
                    "session_id": job.session_id,
                    "compressed_message_id": job.compressed_message_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        finally:
            with self._lock:
                self._threads.discard(current_thread())
            self._slots.release()

    def _write_audit(self, kind: str, payload: dict[str, object]) -> None:
        try:
            CognitionStateStore(self.store).write_audit_record(kind, payload=payload)
        except Exception:
            return


def _report_record(report: WorkerReport) -> dict[str, object]:
    return {
        "worker": report.worker,
        "inspected": report.inspected,
        "emitted": report.emitted,
        "status": report.new_checkpoint.last_status,
        "notes": list(report.notes),
        "yielded": report.yielded_to_higher_priority,
    }


__all__ = ["DirectCompactExtractionService"]
