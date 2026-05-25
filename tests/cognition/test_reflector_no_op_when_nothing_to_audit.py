from __future__ import annotations

from alpha_agent.cognition.reflectors.l1 import ReflectorL1
from tests.cognition.reflector_helpers import context


def test_reflector_no_op_when_nothing_to_audit() -> None:
    assert ReflectorL1().audit(context()) == []
