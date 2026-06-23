from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

import alpha_agent.cognition.state_service as state_service_module
from alpha_agent.cognition.loops.workers.memory_summary import MemorySummaryWorker
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefScope,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    SummaryKind,
    ValidityWindow,
)
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundStage,
)
from alpha_agent.cognition.state_service import CognitionSourceKind, CognitionStateStore
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
    LLMUsage,
)
from alpha_agent.state.store import StateStore


def test_self_memory_summary_worker_writes_validated_summary_with_program_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    first = _self_consolidated_belief(
        "belief:self-root-cause",
        "Agent solves root causes.",
    )
    second = _self_consolidated_belief(
        "belief:self-tests",
        "Agent validates changes with tests.",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json("Agent solves root causes and validates changes with tests."),
        usage=_llm_usage(),
        raw_usage=_raw_llm_usage(),
    )
    processing_time = "2026-06-13T00:00:00+00:00"
    monkeypatch.setattr(state_service_module, "utc_now_iso", lambda: processing_time)

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=2,
        changed_source_min=2,
        invalidated_source_min=1,
    ).run_once()

    assert report.emitted == 1
    summary = service.beliefs.latest_summary(
        summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
        scope=BeliefScope.SELF,
        about=Reference("subject", "subject:self"),
    )
    assert summary is not None
    assert summary.derivation_stage == DerivationStage.BACKGROUND_SUMMARIZED
    assert summary.content == "Agent solves root causes and validates changes with tests."
    expected_source_time = max(str(first.held_since), str(second.held_since))
    assert str(summary.held_since) == expected_source_time
    assert str(summary.validity.observed_at) == expected_source_time
    assert set(summary.source_belief_ids) == {first.id, second.id}
    evidence = {(item.kind, item.id) for item in summary.sources}
    assert any(kind == "background_source_window" for kind, _ in evidence)
    assert ("atomic_belief", str(first.id)) in evidence
    assert ("atomic_belief", str(second.id)) in evidence
    calls = store.list_llm_calls(worker_name="memory_summary")
    assert len(calls) == 1
    assert calls[0].session_id is None
    assert calls[0].provider == provider.name
    assert calls[0].model == "test-summary"
    assert calls[0].total_tokens == 29
    assert calls[0].raw_usage == _raw_llm_usage()


def test_self_memory_summary_worker_uses_only_final_atomic_derivation_stages(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    consolidated = _self_consolidated_belief(
        "belief:self-final-consolidated",
        "Agent solves root causes.",
        derivation_stage=DerivationStage.BACKGROUND_CONSOLIDATED,
    )
    tool_written = _self_consolidated_belief(
        "belief:self-final-tool",
        "Agent uses memory tools deliberately.",
        derivation_stage=DerivationStage.TOOL_WRITTEN,
    )
    human_confirmed = _self_consolidated_belief(
        "belief:self-final-human",
        "Agent should avoid local patches.",
        derivation_stage=DerivationStage.HUMAN_CONFIRMED,
    )
    extracted = _self_consolidated_belief(
        "belief:self-extracted",
        "Extracted self memory is not finalized.",
        derivation_stage=DerivationStage.BACKGROUND_EXTRACTED,
    )
    summarized_stage = _self_consolidated_belief(
        "belief:self-summarized-stage",
        "Summarized-stage atomic records are not summary sources.",
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
    )
    for belief in (consolidated, tool_written, human_confirmed, extracted, summarized_stage):
        service.write_atomic_belief(
            belief,
            source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        )
    provider = _RecordingLLMProvider(
        _summary_json("Agent combines final self memories only.")
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
        summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
        scope=BeliefScope.SELF,
        about=Reference("subject", "subject:self"),
    )
    assert summary is not None
    assert set(summary.source_belief_ids) == {
        consolidated.id,
        tool_written.id,
        human_confirmed.id,
    }
    material = str(provider.calls[0]["messages"][-1]["content"])
    assert str(consolidated.id) in material
    assert str(tool_written.id) in material
    assert str(human_confirmed.id) in material
    assert str(extracted.id) not in material
    assert str(summarized_stage.id) not in material


def test_self_memory_summary_worker_rejects_malformed_llm_output_without_write(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    source = _self_consolidated_belief("belief:self-tests", "Agent validates changes with tests.")
    service.write_atomic_belief(source, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider("{not-json")

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=1,
    ).run_once()

    assert report.emitted == 0
    assert report.new_checkpoint.last_status == "error"
    assert service.beliefs.latest_summary(
        summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
        scope=BeliefScope.SELF,
        about=Reference("subject", "subject:self"),
    ) is None
    window = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)[0]
    assert window.status == BackgroundProgressStatus.FAILED
    assert "malformed" in str(window.last_error)


def test_self_memory_summary_worker_rejects_llm_supplied_provenance_before_write(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    service = CognitionStateStore(store)
    source = _self_consolidated_belief("belief:self-tests", "Agent validates changes with tests.")
    service.write_atomic_belief(source, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    output = json.loads(_summary_json("Agent validates changes with tests."))
    output["payload"]["summary_belief_input"]["source_belief_ids"] = [str(source.id)]
    provider = _RecordingLLMProvider(json.dumps(output, sort_keys=True))

    report = MemorySummaryWorker(
        service,
        provider,
        initial_min_beliefs=1,
    ).run_once()

    assert report.emitted == 0
    assert report.new_checkpoint.last_status == "error"
    assert service.beliefs.latest_summary(
        summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
        scope=BeliefScope.SELF,
        about=Reference("subject", "subject:self"),
    ) is None
    window = service.ledger.list_source_windows(stage=BackgroundStage.SUMMARY)[0]
    assert window.status == BackgroundProgressStatus.FAILED
    assert "source refs" in str(window.last_error)


def test_self_memory_summary_worker_prompt_includes_output_schema_and_target(
    tmp_path,
) -> None:
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
        raw_content="Agent solves root causes.",
        created_at="2026-06-12T01:00:00+00:00",
    )
    second_source = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="Agent validates changes with tests.",
        created_at="2026-06-12T01:17:00+00:00",
    )
    first = _self_consolidated_belief(
        "belief:self-root-cause",
        "Agent solves root causes.",
        sources=[Reference("session_message", first_source.id)],
        held_since="2026-06-12T02:00:00+00:00",
    )
    second = _self_consolidated_belief(
        "belief:self-tests",
        "Agent validates changes with tests.",
        sources=[Reference("session_message", second_source.id)],
        held_since="2026-06-12T02:17:00+00:00",
    )
    service.write_atomic_belief(first, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    service.write_atomic_belief(second, source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS)
    provider = _RecordingLLMProvider(
        _summary_json("Agent solves root causes and validates changes with tests.")
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
    assert '"operation": {' in instruction
    assert '"const": "create_summary_belief"' in instruction
    assert '"const": "skip"' in instruction
    assert '"reason"' in instruction
    assert '"summary_belief_input"' in instruction
    assert '"summary_kind": {' in instruction
    assert '"enum": [' not in instruction
    assert '"const": "self_memory_summary"' in instruction
    assert '"scope": {' in instruction
    assert '"const": "self"' in instruction
    assert '"about": {' in instruction
    assert '{"id": "subject:self", "kind": "subject"}' in instruction
    assert '"structure": {' not in instruction
    assert '"summary_kind": "self_memory_summary"' in material
    assert '{"id": "subject:self", "kind": "subject"}' in material
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


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _self_consolidated_belief(
    belief_id: str,
    content: str,
    *,
    sources: list[Reference] | None = None,
    held_since: str = "2026-01-01T00:00:00+00:00",
    derivation_stage: DerivationStage = DerivationStage.BACKGROUND_CONSOLIDATED,
) -> AtomicBelief:
    return AtomicBelief(
        id=BeliefId(belief_id),
        subject=Reference("subject", "subject:self"),
        about=[Reference("subject", "subject:self")],
        topic="self memory source",
        content=NLStatement(content),
        memory_kind=MemoryKind.FACT,
        derivation_stage=derivation_stage,
        scope=BeliefScope.SELF,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        sources=list(sources or []),
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        held_since=Instant(held_since),
    )


class _RecordingLLMProvider:
    name = "recording-self-summary"

    def __init__(
        self,
        response: str,
        *,
        usage: LLMUsage | None = None,
        raw_usage: dict[str, object] | None = None,
    ) -> None:
        self.response = response
        self.usage = usage
        self.raw_usage = raw_usage
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
        metadata = (
            {"response_payload": {"usage": self.raw_usage}}
            if self.raw_usage is not None
            else {}
        )
        return LLMResponse(
            content=self.response,
            model="test-summary",
            provider=self.name,
            metadata=metadata,
            usage=self.usage,
        )


def _llm_usage() -> LLMUsage:
    return LLMUsage(
        total_tokens=29,
        cached_tokens=5,
        prompt_cache_miss_tokens=17,
        reasoning_tokens=2,
        completion_tokens=7,
    )


def _raw_llm_usage() -> dict[str, object]:
    return {
        "total_tokens": 29,
        "prompt_tokens": 22,
        "prompt_tokens_details": {"cached_tokens": 5},
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


def _summary_json(content: str) -> str:
    return json.dumps(
        {
            "operation": "create_summary_belief",
            "authority": Authority.BACKGROUND_SYNTHESIZED.value,
            "rationale": "Fixture self-memory synthesis.",
            "source_span_note": "from selected self-memory sources",
            "payload": {
                "summary_belief_input": {
                    "summary_kind": SummaryKind.SELF_MEMORY_SUMMARY.value,
                    "scope": BeliefScope.SELF.value,
                    "about": [{"kind": "subject", "id": "subject:self"}],
                    "topic": "self memory summary",
                    "content": content,
                }
            },
        },
        sort_keys=True,
    )
