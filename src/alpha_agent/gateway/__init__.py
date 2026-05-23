"""Platform gateway primitives for Alpha Agent."""

from alpha_agent.gateway.models import (
    ConversationSource,
    DeliveryResult,
    InboundMessage,
    OutboundMessage,
)
from alpha_agent.gateway.runner import (
    ActiveTurnGuard,
    GatewayDeliveryError,
    GatewayRuntimeBridge,
    TurnStartResult,
)
from alpha_agent.gateway.session import (
    DedupResult,
    GatewayDeduplicator,
    GatewaySessionMapping,
    GatewaySessionStore,
    SessionMode,
    generate_session_key,
)

__all__ = [
    "ActiveTurnGuard",
    "ConversationSource",
    "DedupResult",
    "DeliveryResult",
    "GatewayDeduplicator",
    "GatewayDeliveryError",
    "GatewayRuntimeBridge",
    "GatewaySessionMapping",
    "GatewaySessionStore",
    "InboundMessage",
    "OutboundMessage",
    "SessionMode",
    "TurnStartResult",
    "generate_session_key",
]
