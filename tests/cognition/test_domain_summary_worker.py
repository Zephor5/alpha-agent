from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from alpha_agent.cognition.domain_guidance import summary_target_domain
from alpha_agent.cognition.loops.workers.memory_summary import MemorySummaryWorker
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    DerivationStage,
    Instant,
    MemoryKind,
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


def test_domain_summary_worker_writes_llm_synthesized_summary_with_target_identity(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    first = _consolidated_belief(
        "belief:domain-memory-propose-1",
        "Memory proposal direct acceptance applies to remembered preferences.",
        target_domain="memory_propose",
    )
    second = _consolidated_belief(
        "belief:domain-memory-propose-2",
        "Memory proposal direct acceptance applies before changing constraints.",
        target_domain="memory_propose",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal direct acceptance applies.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "direct_accept"},
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
    assert summary.content == "Memory proposal direct acceptance applies."
    assert summary.derivation_stage == DerivationStage.BACKGROUND_SUMMARIZED
    assert summary.structure == {
        "memory_propose": {"policy": "direct_accept"},
        "target_domain": "memory_propose",
    }
    assert summary_target_domain(summary) == "memory_propose"
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


def test_domain_summary_worker_groups_only_final_atomic_derivation_stages(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    consolidated = _consolidated_belief(
        "belief:domain-final-consolidated",
        "Memory proposal acceptance is final after consolidation.",
        target_domain="memory_propose",
        derivation_stage=DerivationStage.BACKGROUND_CONSOLIDATED,
    )
    tool_written = _consolidated_belief(
        "belief:domain-final-tool",
        "Memory proposal tools may write final memory directly.",
        target_domain="memory_propose",
        derivation_stage=DerivationStage.TOOL_WRITTEN,
    )
    human_confirmed = _consolidated_belief(
        "belief:domain-final-human",
        "Human-confirmed memory proposal policy is final.",
        target_domain="memory_propose",
        derivation_stage=DerivationStage.HUMAN_CONFIRMED,
    )
    extracted = _consolidated_belief(
        "belief:domain-extracted",
        "Extracted memory proposal policy is not final.",
        target_domain="memory_propose",
        derivation_stage=DerivationStage.BACKGROUND_EXTRACTED,
    )
    summarized_stage = _consolidated_belief(
        "belief:domain-summarized-stage",
        "Summarized-stage policy is not an atomic summary source.",
        target_domain="memory_propose",
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
    )
    for belief in (consolidated, tool_written, human_confirmed, extracted, summarized_stage):
        service.write_atomic_belief(
            belief,
            source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        )
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal summaries use final atomic sources only.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "final_only"},
            },
        )
    )

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=3,
        changed_source_min=3,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 1
    summary = service.beliefs.latest_summary(
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        scope=BeliefScope.GLOBAL,
    )
    assert summary is not None
    assert set(summary.source_belief_ids) == {
        consolidated.id,
        tool_written.id,
        human_confirmed.id,
    }
    window = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)[0]
    assert window.metadata["summary_target"]["target_domain"] == "memory_propose"
    assert window.metadata["source_belief_ids"] == sorted(
        [str(consolidated.id), str(tool_written.id), str(human_confirmed.id)]
    )
    material = str(provider.calls[0]["messages"][-1]["content"])
    assert str(extracted.id) not in material
    assert str(summarized_stage.id) not in material


def test_domain_summary_worker_runs_invalidated_source_gate(tmp_path) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    old_source = _consolidated_belief(
        "belief:domain-memory-propose-old",
        "Memory proposal direct acceptance applies to obsolete guidance.",
        target_domain="memory_propose",
    )
    current_source = _consolidated_belief(
        "belief:domain-memory-propose-current",
        "Memory proposal direct acceptance applies to current guidance.",
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
            content="Memory proposal direct acceptance applies.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "direct_accept"},
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
    assert latest.content == "Memory proposal direct acceptance applies."


def test_domain_summary_worker_gate_ignores_source_and_holding_time_when_sources_unchanged(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    first = _consolidated_belief(
        "belief:domain-memory-propose-1",
        "Memory proposal direct acceptance applies to remembered preferences.",
        target_domain="memory_propose",
        held_since="2026-06-12T00:00:00+00:00",
    )
    second = _consolidated_belief(
        "belief:domain-memory-propose-2",
        "Memory proposal direct acceptance applies before changing constraints.",
        target_domain="memory_propose",
        held_since="2026-06-12T01:00:00+00:00",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_summary_belief(
        _domain_summary_belief(
            "belief:domain-summary-current",
            source_belief_ids=[first.id, second.id],
        ),
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
    )
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal direct acceptance applies.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "direct_accept"},
            },
        )
    )

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=1,
        changed_source_min=1,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 0
    assert report.new_checkpoint.last_status == "skipped_no_backlog"
    assert provider.calls == []


def test_domain_summary_worker_prompt_includes_target_domain_schema(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session_record(
        "s1",
        timezone="Asia/Shanghai",
        created_at="2026-06-12T00:00:00+00:00",
    )
    service = CognitionStateStore(store)
    first_source = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="Memory proposal direct acceptance applies to remembered preferences.",
        created_at="2026-06-12T01:00:00+00:00",
    )
    second_source = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="Memory proposal direct acceptance applies before changing constraints.",
        created_at="2026-06-12T01:17:00+00:00",
    )
    first = _consolidated_belief(
        "belief:domain-memory-propose-1",
        "Memory proposal direct acceptance applies to remembered preferences.",
        target_domain="memory_propose",
        sources=[Reference("session_message", first_source.id)],
        held_since="2026-06-12T02:00:00+00:00",
    )
    second = _consolidated_belief(
        "belief:domain-memory-propose-2",
        "Memory proposal direct acceptance applies before changing constraints.",
        target_domain="memory_propose",
        sources=[Reference("session_message", second_source.id)],
        held_since="2026-06-12T02:17:00+00:00",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal direct acceptance applies.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "direct_accept"},
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
    messages = provider.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    instruction = messages[1]["content"]
    material = messages[2]["content"]
    assert isinstance(instruction, str)
    assert isinstance(material, str)
    assert '"summary_kind": {' in instruction
    assert '"enum": [' in instruction
    assert '"const": "domain_summary"' not in instruction
    assert '"scope": {' in instruction
    assert '"const": "global"' not in instruction
    assert '"target_domain": {' in instruction
    assert '"const": "memory_propose"' not in instruction
    assert '"summary_kind": "domain_summary"' in material
    assert '"scope": "global"' in material
    assert '"target_domain": "memory_propose"' in material
    assert '"held_since": "2026-06-12T02:00:00+00:00"' in material
    assert '"held_since": "2026-06-12T02:17:00+00:00"' in material
    assert (
        '"source_time_line": "Source message time: 2026-06-12 09:00 '
        '(Asia/Shanghai)."'
    ) in material
    assert (
        '"source_time_line": "Source message time: 2026-06-12 09:17 '
        '(Asia/Shanghai)."'
    ) in material


def test_domain_summary_worker_uses_scope_owner_refs_for_target_identity(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    incidental_entity = Reference("entity", "python")
    first = _consolidated_belief(
        "belief:domain-global-1",
        "Memory proposal direct acceptance applies to Python preferences.",
        target_domain="memory_propose",
        about=[incidental_entity],
    )
    second = _consolidated_belief(
        "belief:domain-global-2",
        "Memory proposal direct acceptance applies to Python constraints.",
        target_domain="memory_propose",
        about=[incidental_entity],
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json(
            summary_kind=SummaryKind.DOMAIN_SUMMARY,
            scope=BeliefScope.GLOBAL,
            about=[],
            content="Memory proposal direct acceptance applies.",
            structure={
                "target_domain": "memory_propose",
                "memory_propose": {"policy": "direct_accept"},
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
    window = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)[0]
    assert window.metadata["summary_target"]["about"] == []
    messages = provider.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    instruction = messages[1]["content"]
    material = messages[2]["content"]
    assert isinstance(instruction, str)
    assert isinstance(material, str)
    assert '"const": []' not in instruction
    assert '"about": []' in material


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _consolidated_belief(
    belief_id: str,
    content: str,
    *,
    target_domain: str,
    scope: BeliefScope = BeliefScope.GLOBAL,
    about: list[Reference] | None = None,
    sources: list[Reference] | None = None,
    held_since: str = "2026-01-01T00:00:00+00:00",
    derivation_stage: DerivationStage = DerivationStage.BACKGROUND_CONSOLIDATED,
) -> AtomicBelief:
    return AtomicBelief(
        id=BeliefId(belief_id),
        subject=Reference("subject", "subject:self"),
        about=list(about or []),
        topic=f"domain guidance {target_domain}",
        content=NLStatement(content),
        memory_kind=MemoryKind.FACT,
        derivation_stage=derivation_stage,
        scope=scope,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        sources=list(sources or []),
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        update_policy={"target_domain": target_domain},
        held_since=Instant(held_since),
    )


def _domain_summary_belief(
    belief_id: str,
    *,
    source_belief_ids: list[BeliefId],
) -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId(belief_id),
        subject=Reference("subject", "subject:self"),
        about=[],
        topic="memory proposal domain guidance",
        content=NLStatement("Old memory proposal direct-accept guidance."),
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=BeliefScope.GLOBAL,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        structure={
            "target_domain": "memory_propose",
            "memory_propose": {"policy": "direct_accept"},
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
            "source_span_note": "from selected consolidated memories",
            "payload": {
                "summary_belief_input": {
                    "summary_kind": summary_kind.value,
                    "scope": scope.value,
                    "about": about,
                    "topic": "memory proposal domain guidance",
                    "content": content,
                    "structure": structure,
                }
            },
        },
        sort_keys=True,
    )
