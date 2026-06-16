import json

import pytest

from alpha_agent.cognition.authority import CognitionSourceKind
from alpha_agent.cognition.background_llm_contract import (
    BackgroundLLMValidationContext,
    BackgroundLLMValidationError,
    SourceWindowValidationContext,
    ValidatedAtomicBeliefDraft,
    ValidatedSummaryBeliefDraft,
    extraction_output_json_schema,
    summary_output_json_schema,
    validate_background_llm_json,
)
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefScope,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Role,
    SituationId,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.processing_ledger import BackgroundSourceRef, BackgroundStage
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.state.store import StateStore


def test_belief_models_expose_validated_topic_not_object() -> None:
    atomic = _atomic_belief(topic="  package management  ")
    summary = _summary_belief(topic="  profile summary  ")

    assert atomic.topic == "package management"
    assert summary.topic == "profile summary"
    assert not hasattr(atomic, "object")
    assert not hasattr(summary, "object")

    atomic_record = atomic.to_record()
    summary_record = summary.to_record()
    assert "topic" in atomic_record
    assert "topic" in summary_record
    assert "object" not in atomic_record
    assert "object" not in summary_record
    assert "structure" not in atomic_record
    assert "structure" in summary_record
    assert "action_orientation" not in atomic_record
    assert "action_orientation" not in summary_record


@pytest.mark.parametrize("topic", ["", "   ", "x" * 65])
def test_belief_models_reject_invalid_topics(topic: str) -> None:
    with pytest.raises(ValueError, match="topic"):
        _atomic_belief(topic=topic)
    with pytest.raises(ValueError, match="topic"):
        _summary_belief(topic=topic)


def test_belief_models_reject_topic_matching_content() -> None:
    with pytest.raises(ValueError, match="topic"):
        _atomic_belief(topic="Python uses indentation.")
    with pytest.raises(ValueError, match="topic"):
        _summary_belief(topic="Python appears in active memories.")


@pytest.mark.parametrize("legacy_key", ["object", "structure", "action_orientation"])
def test_atomic_belief_from_record_rejects_legacy_fields(legacy_key: str) -> None:
    record = _atomic_belief().to_record()
    record[legacy_key] = "legacy"

    with pytest.raises(ValueError, match=legacy_key):
        AtomicBelief.from_record(record)


@pytest.mark.parametrize("legacy_key", ["object", "action_orientation"])
def test_summary_belief_from_record_rejects_legacy_fields(legacy_key: str) -> None:
    record = _summary_belief().to_record()
    record[legacy_key] = "legacy"

    with pytest.raises(ValueError, match=legacy_key):
        SummaryBelief.from_record(record)


def test_background_json_schemas_require_topic_and_drop_legacy_atomic_fields() -> None:
    extraction_schema = extraction_output_json_schema()
    atomic_schema = extraction_schema["properties"]["payload"]["properties"][
        "atomic_belief_inputs"
    ]["items"]
    assert "topic" in atomic_schema["required"]
    assert "topic" in atomic_schema["properties"]
    assert "object" not in atomic_schema["properties"]
    assert "structure" not in atomic_schema["properties"]

    summary_schema = summary_output_json_schema(
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        scope=BeliefScope.GLOBAL,
        about_refs=(),
    )
    summary_create_schema = next(
        branch
        for branch in summary_schema["oneOf"]
        if branch["properties"]["operation"]["const"] == "create_summary_belief"
    )
    summary_skip_schema = next(
        branch
        for branch in summary_schema["oneOf"]
        if branch["properties"]["operation"]["const"] == "skip"
    )
    summary_draft_schema = summary_create_schema["properties"]["payload"]["properties"][
        "summary_belief_input"
    ]
    assert "topic" in summary_draft_schema["required"]
    assert "topic" in summary_draft_schema["properties"]
    assert "object" not in summary_draft_schema["properties"]
    assert "structure" not in summary_draft_schema["properties"]
    assert summary_skip_schema["properties"]["payload"]["required"] == ["reason"]


def test_background_atomic_draft_requires_explicit_valid_topic() -> None:
    validated = validate_background_llm_json(
        _llm_json(
            payload={
                "atomic_belief_inputs": [
                    {
                        "memory_kind": MemoryKind.FACT.value,
                        "scope": BeliefScope.GLOBAL.value,
                        "about": [],
                        "topic": "package management",
                        "content": "Alpha Agent uses uv.",
                    }
                ]
            }
        ),
        _context(),
    )

    draft = validated.payloads[0]
    assert isinstance(draft, ValidatedAtomicBeliefDraft)
    assert draft.topic == "package management"
    assert not hasattr(draft, "object")
    assert not hasattr(draft, "structure")


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({}, "topic"),
        ({"topic": "   "}, "topic"),
        ({"topic": "x" * 65}, "topic"),
        ({"topic": "Alpha Agent uses uv."}, "topic"),
        ({"topic": "package management", "object": "legacy"}, "object"),
        ({"topic": "package management", "structure": {}}, "structure"),
    ],
)
def test_background_atomic_draft_rejects_invalid_topic_or_legacy_fields(
    patch: dict[str, object],
    match: str,
) -> None:
    draft = {
        "memory_kind": MemoryKind.FACT.value,
        "scope": BeliefScope.GLOBAL.value,
        "about": [],
        "content": "Alpha Agent uses uv.",
        **patch,
    }

    with pytest.raises(BackgroundLLMValidationError, match=match):
        validate_background_llm_json(
            _llm_json(payload={"atomic_belief_inputs": [draft]}),
            _context(),
        )


def test_background_summary_draft_requires_explicit_valid_topic() -> None:
    output = _llm_json(
        operation="create_summary_belief",
        payload={
            "summary_belief_input": {
                "summary_kind": SummaryKind.DOMAIN_SUMMARY.value,
                "scope": BeliefScope.GLOBAL.value,
                "about": [],
                "topic": "package management",
                "content": "Alpha Agent package management memories mention uv.",
                "structure": {"target_domain": "package-management"},
            }
        },
    )

    validated = validate_background_llm_json(
        output,
        _context(
            stage=BackgroundStage.SUMMARY,
            allowed_summary_kinds=frozenset({SummaryKind.DOMAIN_SUMMARY}),
            required_summary_scope=BeliefScope.GLOBAL,
            required_summary_target_domain="package-management",
        ),
    )

    draft = validated.payloads[0]
    assert isinstance(draft, ValidatedSummaryBeliefDraft)
    assert draft.topic == "package management"
    assert draft.structure == {"target_domain": "package-management"}


def test_background_non_domain_summary_draft_rejects_structure() -> None:
    output = _llm_json(
        operation="create_summary_belief",
        payload={
            "summary_belief_input": {
                "summary_kind": SummaryKind.SELF_MEMORY_SUMMARY.value,
                "scope": BeliefScope.SELF.value,
                "about": [{"kind": "subject", "id": SUBJECT_SELF}],
                "topic": "answer style",
                "content": "Alpha should answer concisely.",
                "structure": {},
            }
        },
    )

    with pytest.raises(BackgroundLLMValidationError, match="structure"):
        validate_background_llm_json(
            output,
            _context(
                stage=BackgroundStage.SUMMARY,
                allowed_summary_kinds=frozenset({SummaryKind.SELF_MEMORY_SUMMARY}),
                required_summary_scope=BeliefScope.SELF,
            ),
        )


def test_background_non_domain_summary_draft_ignores_null_target_domain_structure() -> None:
    output = _llm_json(
        operation="create_summary_belief",
        payload={
            "summary_belief_input": {
                "summary_kind": SummaryKind.SELF_MEMORY_SUMMARY.value,
                "scope": BeliefScope.SELF.value,
                "about": [{"kind": "subject", "id": SUBJECT_SELF}],
                "topic": "answer style",
                "content": "Alpha should answer concisely.",
                "structure": {"target_domain": None},
            }
        },
    )

    validated = validate_background_llm_json(
        output,
        _context(
            stage=BackgroundStage.SUMMARY,
            allowed_summary_kinds=frozenset({SummaryKind.SELF_MEMORY_SUMMARY}),
            required_summary_scope=BeliefScope.SELF,
        ),
    )

    payload = validated.payloads[0]
    assert isinstance(payload, ValidatedSummaryBeliefDraft)
    assert payload.structure is None


def test_background_output_rejects_update_policy_summary_targets_by_default() -> None:
    draft = {
        "memory_kind": MemoryKind.FACT.value,
        "scope": BeliefScope.GLOBAL.value,
        "about": [],
        "topic": "package management",
        "content": "Alpha Agent uses uv.",
        "update_policy": {"target_domain": "package-management"},
    }

    with pytest.raises(BackgroundLLMValidationError, match="target_domain"):
        validate_background_llm_json(
            _llm_json(payload={"atomic_belief_inputs": [draft]}),
            _context(),
        )

    validated = validate_background_llm_json(
        _llm_json(payload={"atomic_belief_inputs": [draft]}),
        _context(allow_summary_scheduling_hints=True),
    )
    payload = validated.payloads[0]
    assert isinstance(payload, ValidatedAtomicBeliefDraft)
    assert payload.update_policy == {"target_domain": "package-management"}


def test_state_service_persists_topic_without_object_columns(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    service = CognitionStateStore(store)
    belief = _atomic_belief(topic="package management")

    written = service.write_atomic_belief(
        belief,
        source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
    )
    retrieved = service.beliefs.get_by_id(written.id)

    assert retrieved == belief
    assert isinstance(retrieved, AtomicBelief)
    assert retrieved.topic == "package management"

    with store.connect() as conn:
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(atomic_beliefs)")]
        row = conn.execute(
            "SELECT topic, record FROM atomic_beliefs WHERE id = ?",
            (str(belief.id),),
        ).fetchone()

    assert "topic" in columns
    assert "object" not in columns
    assert row["topic"] == "package management"
    record = json.loads(row["record"])
    assert record["topic"] == "package management"
    assert "object" not in record


def test_state_service_accepts_background_topic_without_content_fallback(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    service = CognitionStateStore(store)
    source = BackgroundSourceRef("session_message", "msg-1")
    window = service.ledger.create_source_window(
        stage=BackgroundStage.EXTRACTION,
        target_unit="session:s1",
        source_refs=(source,),
        idempotency_key="extract:s1:topic",
    )

    accepted = service.accept_background_llm_json(
        _llm_json(
            payload={
                "atomic_belief_inputs": [
                    {
                        "memory_kind": MemoryKind.FACT.value,
                        "scope": BeliefScope.GLOBAL.value,
                        "about": [],
                        "topic": "package management",
                        "content": "Alpha Agent uses uv.",
                    }
                ]
            }
        ),
        _context(window_id=window.window_id, source_refs=(source,)),
        window_id=window.window_id,
        run_id=None,
        checkpoint_id="checkpoint:topic",
    )

    assert len(accepted) == 1
    assert isinstance(accepted[0], AtomicBelief)
    assert accepted[0].topic == "package management"


def _atomic_belief(*, topic: str = "python") -> AtomicBelief:
    return AtomicBelief(
        id=BeliefId("belief:test"),
        subject=subject_ref(SUBJECT_SELF),
        about=[],
        topic=topic,
        content=NLStatement("Python uses indentation."),
        memory_kind=MemoryKind.FACT,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope.GLOBAL,
        authority=Authority.USER_ASSERTED,
        sources=[],
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        relations=[],
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


def _summary_belief(*, topic: str = "profile") -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId("belief:summary"),
        subject=subject_ref(SUBJECT_SELF),
        about=[],
        topic=topic,
        content=NLStatement("Python appears in active memories."),
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=BeliefScope.GLOBAL,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        structure={"target_domain": "profile"},
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


def _context(
    *,
    window_id: str = "window:test",
    source_refs: tuple[BackgroundSourceRef, ...] = (
        BackgroundSourceRef("session_message", "msg-1"),
    ),
    stage: BackgroundStage = BackgroundStage.EXTRACTION,
    allowed_summary_kinds: frozenset[SummaryKind] | None = None,
    required_summary_scope: BeliefScope | None = None,
    required_summary_target_domain: str | None = None,
    allow_summary_scheduling_hints: bool = False,
) -> BackgroundLLMValidationContext:
    return BackgroundLLMValidationContext(
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        source_window=SourceWindowValidationContext(
            window_id=window_id,
            source_refs=source_refs,
            stage=stage,
            target_unit="session:s1",
            session_id="s1",
        ),
        allowed_summary_kinds=allowed_summary_kinds,
        required_summary_scope=required_summary_scope,
        required_summary_target_domain=required_summary_target_domain,
        allow_summary_scheduling_hints=allow_summary_scheduling_hints,
    )


def _llm_json(
    *,
    operation: str = "create_atomic_belief",
    payload: dict[str, object],
) -> str:
    return json.dumps(
        {
            "operation": operation,
            "authority": Authority.BACKGROUND_SYNTHESIZED.value,
            "rationale": "Fixture rationale.",
            "payload": payload,
        },
        sort_keys=True,
    )
