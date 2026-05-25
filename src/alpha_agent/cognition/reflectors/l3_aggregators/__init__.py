"""Deterministic SelfModel aggregators for Reflector L3."""

from alpha_agent.cognition.reflectors.l3_aggregators.capabilities_aggregator import (
    CapabilitiesAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.failure_modes_aggregator import (
    FailureModesAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.interaction_patterns_aggregator import (
    InteractionPatternsAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.preferred_strategies_aggregator import (
    PreferredStrategiesAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import (
    AggregationWindow,
    SelfModelAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.stable_preferences_aggregator import (
    StablePreferencesAggregator,
)
from alpha_agent.cognition.reflectors.l3_aggregators.tradeoff_aggregator import (
    TradeoffAggregator,
)


def default_aggregators() -> list[SelfModelAggregator]:
    return [
        CapabilitiesAggregator(),
        FailureModesAggregator(),
        PreferredStrategiesAggregator(),
        StablePreferencesAggregator(),
        TradeoffAggregator(),
        InteractionPatternsAggregator(),
    ]


__all__ = [
    "AggregationWindow",
    "CapabilitiesAggregator",
    "FailureModesAggregator",
    "InteractionPatternsAggregator",
    "PreferredStrategiesAggregator",
    "SelfModelAggregator",
    "StablePreferencesAggregator",
    "TradeoffAggregator",
    "default_aggregators",
]
