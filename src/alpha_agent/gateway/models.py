"""Platform-neutral gateway message models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from alpha_agent.utils.time import utc_now

ChatType = Literal["dm", "group", "channel"]
MessageType = Literal["text", "command", "attachment", "system"]
Visibility = Literal["default", "public", "ephemeral"]

Attachment = dict[str, Any]
Metadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConversationSource:
    """External platform identity for a single inbound conversation location."""

    platform: str
    chat_id: str
    chat_type: ChatType | str
    user_id: str
    user_name: str | None = None
    thread_id: str | None = None
    message_id: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """A normalized inbound message delivered by a platform adapter."""

    source: ConversationSource
    text: str
    message_type: MessageType | str = "text"
    attachments: list[Attachment] = field(default_factory=list)
    received_at: datetime = field(default_factory=utc_now)
    platform_message_id: str | None = None
    raw_metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """A normalized response requested by the Alpha runtime."""

    text: str
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: str | None = None
    thread_metadata: Metadata = field(default_factory=dict)
    visibility: Visibility = "default"


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Result returned by a platform adapter after sending a message."""

    success: bool
    message_id: str | None = None
    error: str | None = None
    retryable: bool = False
