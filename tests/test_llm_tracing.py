from __future__ import annotations

from pathlib import Path

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.tracing import LLMTraceLogger


def test_llm_trace_logger_from_config_uses_runtime_debug_flag(tmp_path: Path) -> None:
    disabled = AlphaConfig(
        db_path=tmp_path / "disabled.db",
        log_dir=tmp_path / "disabled-logs",
        gateway_status_path=tmp_path / "disabled-gateway.json",
        llm_debug_logging=False,
    )
    enabled = AlphaConfig(
        db_path=tmp_path / "enabled.db",
        log_dir=tmp_path / "enabled-logs",
        gateway_status_path=tmp_path / "enabled-gateway.json",
        llm_debug_logging=True,
    )

    assert not LLMTraceLogger.from_config(disabled).enabled
    logger = LLMTraceLogger.from_config(enabled)
    assert logger.enabled
    assert logger.trace_log_path == tmp_path / "enabled-logs" / "llm.jsonl"
