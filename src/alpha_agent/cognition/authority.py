"""Authority ceiling rules for cognition state writes."""

from __future__ import annotations

from enum import StrEnum

from alpha_agent.cognition.models import Authority


class CognitionSourceKind(StrEnum):
    """Program-owned source categories used to bound accepted belief authority."""

    SYSTEM_PROJECT_RULE = "system_project_rule"
    EXPLICIT_CONFIRMATION_FLOW = "explicit_confirmation_flow"
    DIRECT_USER_STATEMENT = "direct_user_statement"
    BACKGROUND_SYNTHESIS = "background_synthesis"
    LLM_INTERPRETATION = "llm_interpretation"


class AuthorityOverclaimError(ValueError):
    """Raised when a proposed belief authority exceeds its source ceiling."""


_AUTHORITY_RANK: dict[Authority, int] = {
    Authority.LLM_INTERPRETED: 1,
    Authority.BACKGROUND_SYNTHESIZED: 2,
    Authority.USER_ASSERTED: 3,
    Authority.HUMAN_CONFIRMED: 4,
    Authority.SYSTEM_DEFINED: 5,
}

_SOURCE_AUTHORITY_CEILING: dict[CognitionSourceKind, Authority] = {
    CognitionSourceKind.SYSTEM_PROJECT_RULE: Authority.SYSTEM_DEFINED,
    CognitionSourceKind.EXPLICIT_CONFIRMATION_FLOW: Authority.HUMAN_CONFIRMED,
    CognitionSourceKind.DIRECT_USER_STATEMENT: Authority.USER_ASSERTED,
    CognitionSourceKind.BACKGROUND_SYNTHESIS: Authority.BACKGROUND_SYNTHESIZED,
    CognitionSourceKind.LLM_INTERPRETATION: Authority.LLM_INTERPRETED,
}


def authority_ceiling(source_kind: CognitionSourceKind | str) -> Authority:
    """Return the highest authority accepted for a program-owned source kind."""

    return _SOURCE_AUTHORITY_CEILING[CognitionSourceKind(source_kind)]


def authority_allows(proposed: Authority | str, *, source_kind: CognitionSourceKind | str) -> bool:
    """Return whether ``proposed`` is at or below the source authority ceiling."""

    proposed_authority = Authority(proposed)
    ceiling = authority_ceiling(source_kind)
    return _AUTHORITY_RANK[proposed_authority] <= _AUTHORITY_RANK[ceiling]


def require_authority_within_ceiling(
    proposed: Authority | str,
    *,
    source_kind: CognitionSourceKind | str,
) -> Authority:
    """Validate and return authority, rejecting source overclaims."""

    proposed_authority = Authority(proposed)
    if authority_allows(proposed_authority, source_kind=source_kind):
        return proposed_authority
    ceiling = authority_ceiling(source_kind)
    raise AuthorityOverclaimError(
        f"authority {proposed_authority.value!r} exceeds "
        f"{CognitionSourceKind(source_kind).value!r} ceiling {ceiling.value!r}"
    )
