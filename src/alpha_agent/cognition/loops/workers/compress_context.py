"""Compress old foreground context into deterministic background summaries."""

from __future__ import annotations

import json
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import ScheduleTrigger, WorkerCheckpoint, WorkerReport
from alpha_agent.cognition.loops.workers._common import (
    after_cursor_wrap,
    emit_projected,
    report,
    stable_id,
    trigger,
)
from alpha_agent.cognition.models import CognitiveEventKind, Subject
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class CompressContextWorker:
    name: ClassVar[str] = "compress_context"
    trigger: ClassVar[ScheduleTrigger] = trigger(10, 6, {CognitiveEventKind.PERCEIVED}, 12)
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.PERCEIVED, CognitiveEventKind.CONTEXT_COMPRESSED}
    )

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: object,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        del log
        projection = projections.get_typed(ContextWindowProjection)
        max_foreground = max(1, int(getattr(config, "context_foreground_max", 8)))
        absorb_batch = max(1, int(getattr(config, "context_absorb_batch", 4)))
        emitted = 0
        inspected = 0
        thread_ids = sorted(projection.list_thread_ids(), key=_thread_cursor)
        pending = after_cursor_wrap(
            thread_ids,
            str(checkpoint.metadata.get("last_thread_id", "")),
            _thread_cursor,
        )
        for thread_id in pending:
            inspected += 1
            foreground_ids = projection.foreground_ids(thread_id)
            if len(foreground_ids) > max_foreground:
                window = projection.get(thread_id, Subject())
                anchored_ids = set(window.metadata.get("anchored_ids", []))
                absorbable = [item for item in foreground_ids if item not in anchored_ids]
                take = min(absorb_batch, max(0, len(foreground_ids) - max_foreground))
                absorbed_ids = absorbable[:take]
                if absorbed_ids:
                    absorbed_text = [
                        perception.raw
                        for perception in window.foreground
                        if str(perception.id) in set(absorbed_ids)
                    ]
                    summary = _summary(
                        absorbed_text,
                        int(getattr(config, "context_summary_chars", 480)),
                    )
                    summary_id = stable_id("ctxbg", thread_id.to_record(), absorbed_ids, summary)
                    event = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.CONTEXT_COMPRESSED,
                        config=config,
                        payload={
                            "thread_id": thread_id.to_record(),
                            "absorbed_perception_ids": absorbed_ids,
                            "produced_summary_id": summary_id,
                            "background_summary_id": summary_id,
                            "summary": summary,
                            "compression_policy": "deterministic_v1",
                            "preserved_anchors": sorted(anchored_ids),
                        },
                        rationale=(
                            "Compressed old foreground context into deterministic background."
                        ),
                    )
                    emitted += (
                        1 if event is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=inspected,
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_thread_id": _thread_cursor(thread_id)},
                )
        return report(self.name, checkpoint, inspected=inspected, emitted=emitted, metadata={})


def _summary(lines: list[str], limit: int) -> str:
    joined = " | ".join(item.strip() for item in lines if item.strip())
    if len(joined) <= limit:
        return joined
    return joined[: max(0, limit - 3)].rstrip() + "..."


def _thread_cursor(thread_id) -> str:
    return json.dumps(thread_id.to_record(), ensure_ascii=False, sort_keys=True)
