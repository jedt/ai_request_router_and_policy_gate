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
    return (
        profile.reasoning_depth * provider.reasoning_score * policy.reasoning_weight
        + profile.latency_sensitivity * provider.latency_score * policy.latency_weight
        + profile.cost_sensitivity * provider.cost_score * policy.cost_weight
    )


def rank_providers(
    profile: QueryProfile,
    providers: Sequence[ProviderProfile],
    policy: RoutingPolicy,
) -> tuple[ProviderProfile, ...]:
    """Rank providers by suitability with a deterministic ID tie breaker."""
    return tuple(
        sorted(
            providers,
            key=lambda provider: (
                -provider_score(profile, provider, policy),
                provider.id,
            ),
        )
    )


def select_provider(
    profile: QueryProfile,
    providers: Sequence[ProviderProfile],
    policy: RoutingPolicy,
) -> RoutingDecision:

    if not providers:
        raise ValueError("No providers configured.")

    winner = rank_providers(profile, providers, policy)[0]

    return RoutingDecision(
        provider=winner,
        profile=profile,
        eligible_provider_ids=tuple(provider.id for provider in providers),
        score=provider_score(profile, winner, policy),
    )
