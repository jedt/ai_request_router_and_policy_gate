from __future__ import annotations

from llm_router.models import QueryProfile


def simulate_low_reasoning_user_query_classification() -> QueryProfile:
    return QueryProfile(
        reasoning_depth=0.05,
        latency_sensitivity=0.95,
        cost_sensitivity=0.95,
    )


def simulate_medium_reasoning_user_query_classification() -> QueryProfile:
    return QueryProfile(
        reasoning_depth=0.50,
        latency_sensitivity=0.60,
        cost_sensitivity=0.60,
    )


def simulate_high_reasoning_user_query_classification() -> QueryProfile:
    return QueryProfile(
        reasoning_depth=0.95,
        latency_sensitivity=0.10,
        cost_sensitivity=0.10,
    )