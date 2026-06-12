"""Shared system-reminder marker constants."""

from __future__ import annotations

SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"
SYSTEM_REMINDER_PLACEHOLDER = f"{SYSTEM_REMINDER_OPEN}...{SYSTEM_REMINDER_CLOSE}"


def inline_system_reminder(content: str) -> str:
    """Wrap reminder content without adding internal newlines."""

    return f"{SYSTEM_REMINDER_OPEN}{content.strip()}{SYSTEM_REMINDER_CLOSE}"
