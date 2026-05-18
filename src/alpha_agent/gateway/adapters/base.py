"""Framework-free platform adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from alpha_agent.gateway.models import (
    ConversationSource,
    DeliveryResult,
    InboundMessage,
    OutboundMessage,
)

InboundHandler = Callable[[InboundMessage], OutboundMessage | None]


class PlatformAdapter(ABC):
    """Minimal sync interface implemented by concrete platform gateways."""

    @abstractmethod
    def connect(self, handler: InboundHandler) -> None:
        """Connect to the platform and dispatch inbound messages to handler."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the platform."""

    @abstractmethod
    def send(self, source: ConversationSource, outbound: OutboundMessage) -> DeliveryResult:
        """Send an outbound message to the platform conversation."""

    @abstractmethod
    def send_typing(self, source: ConversationSource) -> None:
        """Signal typing or equivalent processing state where supported."""

    def on_processing_start(self, source: ConversationSource) -> None:
        """Optional hook called before Alpha begins processing an inbound turn."""
        return None

    def on_processing_complete(self, source: ConversationSource) -> None:
        """Optional hook called after Alpha finishes processing an inbound turn."""
        return None
