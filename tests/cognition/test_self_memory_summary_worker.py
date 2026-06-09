from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from alpha_agent.cognition.loops.workers.memory_summary import MemorySummaryWorker
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefScope,
    DerivationStage,
    Reference,
    SummaryKind,
)
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundStage,
)
from alpha_agent.cognition.state_service import CognitionSourceKind, CognitionStateStore
from alpha_agent.llm.base import ChatMessage, LLMResponse, LLMToolChoice, LLMToolDefinitionInput
from alpha_agent.state.store import StateStore
from tests.cognition.test_belief_projection_apply import belief


def test_self_memory_summary_worker_writes_validated_summary_with_program_sources(
    tmp_path,
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
    summary = service.beliefs.latest_summary(
        summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
        scope=BeliefScope.SELF,
        about=Reference("subject", "subject:self"),
    )
    assert summary is not None
    assert summary.derivation_stage == DerivationStage.BACKGROUND_SUMMARIZED
    assert summary.content == "Agent solves root causes and validates changes with tests."
    assert set(summary.source_belief_ids) == {first.id, second.id}
    evidence = {(item.kind, item.id) for item in summary.sources}
    assert any(kind == "background_source_window" for kind, _ in evidence)
    assert ("atomic_belief", str(first.id)) in evidence
    assert ("atomic_belief", str(second.id)) in evidence


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
    output["payload"]["summary_belief_draft"]["source_belief_ids"] = [str(source.id)]
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
    instruction = provider.calls[0]["messages"][0]["content"]
    assert isinstance(instruction, str)
    assert '"operation": {' in instruction
    assert '"const": "create_summary_belief"' in instruction
    assert '"summary_belief_draft"' in instruction
    assert '"summary_kind": {' in instruction
    assert '"const": "self_memory_summary"' in instruction
    assert '"scope": {' in instruction
    assert '"const": "self"' in instruction
    assert '"about": {' in instruction
    assert '{"id": "subject:self", "kind": "subject"}' in instruction


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _self_consolidated_belief(belief_id: str, content: str) -> AtomicBelief:
    record = belief(
        belief_id,
        content,
        about=[Reference("subject", "subject:self")],
        object_="self memory source",
        scope=BeliefScope.SELF,
    ).to_record()
    record["authority"] = Authority.BACKGROUND_SYNTHESIZED.value
    record["derivation_stage"] = DerivationStage.BACKGROUND_CONSOLIDATED.value
    return AtomicBelief.from_record(record)


class _RecordingLLMProvider:
    name = "recording-self-summary"

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


def _summary_json(content: str) -> str:
    return json.dumps(
        {
            "operation": "create_summary_belief",
            "authority": Authority.BACKGROUND_SYNTHESIZED.value,
            "rationale": "Fixture self-memory synthesis.",
            "requires_confirmation": False,
            "source_span_note": "from selected self-memory sources",
            "payload": {
                "summary_belief_draft": {
                    "summary_kind": SummaryKind.SELF_MEMORY_SUMMARY.value,
                    "scope": BeliefScope.SELF.value,
                    "about": [{"kind": "subject", "id": "subject:self"}],
                    "object": "self memory summary",
                    "content": content,
                    "structure": {},
                }
            },
        },
        sort_keys=True,
    )
