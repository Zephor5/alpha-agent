"""Runtime serialization helpers."""

from __future__ import annotations

import json
from typing import Any


def deterministic_json(value: Any) -> str:
    """Serialize event payloads deterministically for replayable logs."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
