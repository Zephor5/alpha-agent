"""Shared payload helpers for reactive stages."""

from __future__ import annotations

import hashlib
from typing import Any

from alpha_agent.cognition.models import Reference


def digest_payload(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def ref_ids(refs: list[Reference]) -> list[str]:
    return [ref.id for ref in refs]
