"""Interpret stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    Belief,
    BeliefId,
    BeliefRef,
    CognitiveEventKind,
    ContextWindow,
    EventId,
    NLStatement,
    Reference,
    Subject,
    belief_ref,
)
from alpha_agent.cognition.stages._payload import ref_ids
from alpha_agent.cognition.stages.types import AttentionFocus, Emitted, Interpretation
from alpha_agent.cognition.value.resolver import resolve_conflict


class Interpreter:
    """Compare the focus against recalled beliefs."""

    def interpret(
        self,
        focus: AttentionFocus,
        window: ContextWindow,
        recalled: list[Belief | BeliefRef],
        subject: Subject,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[Interpretation]:
        text = "\n".join(str(claim) for claim in focus.salient_claims)
        recalled_beliefs = [item for item in recalled if isinstance(item, Belief)]
        recalled_refs = [_belief_reference(item) for item in recalled]
        support_refs: list[BeliefRef] = []
        contradict_refs: list[BeliefRef] = []
        claim_texts = [_normalize(claim) for claim in focus.salient_claims if str(claim).strip()]
        for belief in recalled_beliefs:
            belief_content = _normalize(belief.content)
            if belief_content in claim_texts:
                support_refs.append(belief_ref(belief.id))
                continue
            if any(_same_relation_different_object(claim, belief_content) for claim in claim_texts):
                contradict_refs.append(belief_ref(belief.id))
        proposed_resolution = None
        recalled_conflict = _first_recalled_conflict(recalled_beliefs)
        if recalled_conflict is not None:
            left, right = recalled_conflict
            proposed_resolution = resolve_conflict(left, right, subject.value_lens)
            contradict_refs.extend([belief_ref(left.id), belief_ref(right.id)])
        if support_refs:
            stance = "consistent"
        elif contradict_refs or proposed_resolution is not None:
            stance = "contradicting"
        elif not claim_texts:
            stance = "ambiguous"
        elif not recalled:
            stance = "novel"
        elif not recalled_beliefs:
            stance = "consistent"
            support_refs = recalled_refs
        else:
            stance = "ambiguous"
        interpretation = Interpretation(
            stance=stance,
            supporting_beliefs=support_refs,
            contradicting_beliefs=contradict_refs,
            novel_claims=list(focus.salient_claims) if stance == "novel" else [],
            ambiguity_notes=[] if text else ["empty stimulus"],
            source_text=text,
            proposed_resolution=proposed_resolution,
        )
        resolution_payload = (
            proposed_resolution.to_payload() if proposed_resolution is not None else None
        )
        event = emitter.emit(
            CognitiveEventKind.INTERPRETED,
            situation=window.situation_at,
            inputs=[Reference("subject", str(subject.id))],
            rationale=NLStatement("Interpreted focus against recalled beliefs."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "stance": interpretation.stance,
                "support_ids": ref_ids(interpretation.supporting_beliefs),
                "contradict_ids": ref_ids(interpretation.contradicting_beliefs),
                "novel_claim_count": len(interpretation.novel_claims),
                "proposed_resolution": resolution_payload,
            },
        )
        return Emitted(interpretation, event)


def _belief_reference(value: Belief | BeliefRef) -> BeliefRef:
    if isinstance(value, Belief):
        return belief_ref(BeliefId(str(value.id)))
    return value


def _normalize(value: object) -> str:
    return " ".join(str(value).casefold().strip().split())


def _same_relation_different_object(claim: str, belief: str) -> bool:
    claim_parts = _statement_parts(claim)
    belief_parts = _statement_parts(belief)
    if claim_parts is None or belief_parts is None:
        return False
    return claim_parts[:2] == belief_parts[:2] and claim_parts[2] != belief_parts[2]


def _first_recalled_conflict(beliefs: list[Belief]) -> tuple[Belief, Belief] | None:
    for left_index, left in enumerate(beliefs):
        left_content = _normalize(left.content)
        for right in beliefs[left_index + 1:]:
            if _same_relation_different_object(left_content, _normalize(right.content)):
                return left, right
    return None


def _statement_parts(value: str) -> tuple[str, str, str] | None:
    words = value.rstrip(".").split()
    if len(words) < 3:
        return None
    return words[0], words[1], " ".join(words[2:])
