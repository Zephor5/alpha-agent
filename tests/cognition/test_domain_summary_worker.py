from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from alpha_agent.cognition.loops.workers.memory_summary import MemorySummaryWorker
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    DerivationStage,
    Instant,
    NLStatement,
    Reference,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
)
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundStage,
)
from alpha_agent.cognition.state_service import CognitionSourceKind, CognitionStateStore
from alpha_agent.llm.base import ChatMessage, LLMResponse, LLMToolChoice, LLMToolDefinitionInput
from alpha_agent.state.store import StateStore
from tests.cognition.test_belief_projection_apply import belief


def test_domain_summary_worker_writes_llm_synthesized_summary_with_target_identity(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    first = _consolidated_belief(
        "belief:domain-memory-propose-1",
        "Memory proposal confirmation required for remembered preferences.",
        target_domain="memory_propose",
    )
    second = _consolidated_belief(
        "belief:domain-memory-propose-2",
        "Memory proposal confirmation required before changing constraints.",
        target_domain="memory_propose",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal confirmation required.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"requires_confirmation": True},
            },
        )
    )

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=2,
        changed_source_min=2,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 1
    assert len(provider.calls) == 1
    summary = service.beliefs.latest_summary(
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        scope=BeliefScope.GLOBAL,
    )
    assert summary is not None
    assert summary.content == "Memory proposal confirmation required."
    assert summary.derivation_stage == DerivationStage.BACKGROUND_SUMMARIZED
    assert summary.structure == {
        "memory_propose": {"requires_confirmation": True},
        "target_domain": "memory_propose",
    }
    assert set(summary.source_belief_ids) == {first.id, second.id}
    windows = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)
    assert len(windows) == 1
    assert windows[0].status == BackgroundProgressStatus.PROCESSED
    assert windows[0].metadata["summary_target"] == {
        "about": [],
        "scope": "global",
        "summary_kind": "domain_summary",
        "target_domain": "memory_propose",
    }


def test_domain_summary_worker_runs_invalidated_source_gate(tmp_path) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    old_source = _consolidated_belief(
        "belief:domain-memory-propose-old",
        "Memory proposal confirmation required for obsolete guidance.",
        target_domain="memory_propose",
    )
    current_source = _consolidated_belief(
        "belief:domain-memory-propose-current",
        "Memory proposal confirmation required for current guidance.",
        target_domain="memory_propose",
    )
    service.write_atomic_belief(old_source, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_summary_belief(
        _domain_summary_belief(
            "belief:domain-summary-old",
            source_belief_ids=[old_source.id],
        ),
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
    )
    service.mark_belief_lifecycle(
        old_source.id,
        BeliefLifecycle.ARCHIVED,
        at="2026-01-02T00:00:00+00:00",
    )
    service.write_atomic_belief(
        current_source,
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
    )
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal confirmation required.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"requires_confirmation": True},
            },
        )
    )

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=99,
        changed_source_min=99,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 1
    window = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)[0]
    assert window.metadata["gate"] == "invalidated_source"
    latest = service.beliefs.latest_summary(
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        scope=BeliefScope.GLOBAL,
    )
    assert latest is not None
    assert latest.content == "Memory proposal confirmation required."


def test_domain_summary_worker_prompt_includes_target_domain_schema(tmp_path) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    first = _consolidated_belief(
        "belief:domain-memory-propose-1",
        "Memory proposal confirmation required for remembered preferences.",
        target_domain="memory_propose",
    )
    second = _consolidated_belief(
        "belief:domain-memory-propose-2",
        "Memory proposal confirmation required before changing constraints.",
        target_domain="memory_propose",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal confirmation required.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"requires_confirmation": True},
            },
        )
    )

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=2,
        changed_source_min=2,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 1
    instruction = provider.calls[0]["messages"][0]["content"]
    assert isinstance(instruction, str)
    assert '"summary_kind": {' in instruction
    assert '"const": "domain_summary"' in instruction
    assert '"scope": {' in instruction
    assert '"const": "global"' in instruction
    assert '"target_domain": {' in instruction
    assert '"const": "memory_propose"' in instruction


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _consolidated_belief(
    belief_id: str,
    content: str,
    *,
    target_domain: str,
) -> AtomicBelief:
    record = belief(
        belief_id,
        content,
        about=[],
        object_=f"domain guidance {target_domain}",
    ).to_record()
    record["authority"] = Authority.BACKGROUND_SYNTHESIZED.value
    record["derivation_stage"] = DerivationStage.BACKGROUND_CONSOLIDATED.value
    record["structure"] = {"target_domain": target_domain}
    return AtomicBelief.from_record(record)


def _domain_summary_belief(
    belief_id: str,
    *,
    source_belief_ids: list[BeliefId],
) -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId(belief_id),
        subject=Reference("subject", "subject:self"),
        about=[],
        object="memory proposal domain guidance",
        content=NLStatement("Old memory proposal confirmation guidance."),
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=BeliefScope.GLOBAL,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        structure={
            "target_domain": "memory_propose",
            "memory_propose": {"requires_confirmation": True},
        },
        source_belief_ids=source_belief_ids,
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


class _RecordingLLMProvider:
    name = "recording-domain-summary"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: object | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        return LLMResponse(content=self.response, model="test-summary", provider=self.name)


def _summary_json(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about: list[dict[str, str]],
    content: str,
    structure: dict[str, object],
) -> str:
    return json.dumps(
        {
            "operation": "create_summary_belief",
            "authority": Authority.BACKGROUND_SYNTHESIZED.value,
            "rationale": "Fixture domain guidance synthesis.",
            "requires_confirmation": False,
            "source_span_note": "from selected consolidated memories",
            "payload": {
                "summary_belief_draft": {
                    "summary_kind": summary_kind.value,
                    "scope": scope.value,
                    "about": about,
                    "object": "memory proposal domain guidance",
                    "content": content,
                    "structure": structure,
                }
            },
        },
        sort_keys=True,
    )
