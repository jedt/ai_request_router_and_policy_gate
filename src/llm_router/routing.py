from __future__ import annotations

from collections.abc import Sequence

from llm_router.models import (
    ProviderProfile,
    QueryProfile,
    RoutingDecision,
    RoutingPolicy,
)


def provider_score(
    profile: QueryProfile,
    provider: ProviderProfile,
    policy: RoutingPolicy,
) -> float:
    """Calculate provider suitability from requirements and capability scores."""
    return (
        profile.reasoning_depth * provider.reasoning_score * policy.reasoning_weight
        + profile.latency_sensitivity * provider.latency_score * policy.latency_weight
        + profile.cost_sensitivity * provider.cost_score * policy.cost_weight
    )


def select_provider(
    profile: QueryProfile,
    providers: Sequence[ProviderProfile],
    policy: RoutingPolicy,
) -> RoutingDecision:
    """Select the highest-scoring provider with a deterministic tie-break."""
    if not providers:
        raise ValueError("No providers configured.")

    winner = min(
        providers,
        key=lambda provider: (-provider_score(profile, provider, policy), provider.id),
    )

    return RoutingDecision(
        provider=winner,
        profile=profile,
        eligible_provider_ids=tuple(provider.id for provider in providers),
        score=provider_score(profile, winner, policy),
    )
