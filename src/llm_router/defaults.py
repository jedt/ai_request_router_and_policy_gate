from __future__ import annotations

from pathlib import Path

from llm_router.models import ProviderProfile, RoutingPolicy


DEFAULT_CLASSIFIER_MODEL = "gpt-4.1-nano"
DEFAULT_APPROVAL_POLICY_PATH = Path(__file__).resolve().parents[2] / "approval-policy.json"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0

DEFAULT_PROVIDERS: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        id="mock-fast",
        provider="mock-provider-a",
        model="gpt-4o-mini",
        reasoning_score=0.45,
        latency_score=0.98,
        cost_score=0.98,
    ),
    ProviderProfile(
        id="mock-balanced",
        provider="mock-provider-b",
        model="gpt-4.1-mini",
        reasoning_score=0.75,
        latency_score=0.75,
        cost_score=0.70,
    ),
    ProviderProfile(
        id="mock-reasoning",
        provider="mock-provider-c",
        model="gpt-5",
        reasoning_score=0.99,
        latency_score=0.35,
        cost_score=0.25,
    ),
)

DEFAULT_POLICY = RoutingPolicy()

FAILOVER_TEST_PROVIDERS: tuple[ProviderProfile, ...] = (
    ProviderProfile(
        id="openai-fast",
        provider="openai",
        model="gpt-4o-mini",
        reasoning_score=0.85,
        latency_score=0.98,
        cost_score=0.90,
    ),
    ProviderProfile(
        id="gemini-flash",
        provider="google",
        model="gemini-2.5-flash",
        reasoning_score=0.80,
        latency_score=0.90,
        cost_score=0.85,
    ),
)
