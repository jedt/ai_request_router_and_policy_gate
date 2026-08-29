from __future__ import annotations

import unittest

import pytest

from llm_router.defaults import DEFAULT_POLICY, DEFAULT_PROVIDERS
from llm_router.models import ProviderProfile, QueryProfile, RoutingPolicy
from llm_router.routing import select_provider


def test_select_provider_returns_complete_routing_decision() -> None:
    profile = QueryProfile(
        reasoning_depth=1,
        latency_sensitivity=0,
        cost_sensitivity=0,
    )
    policy = RoutingPolicy(
        reasoning_weight=1,
        latency_weight=0,
        cost_weight=0,
    )
    providers = (
        ProviderProfile("basic", "vendor-a", "model-a", 0.2, 1, 1),
        ProviderProfile("reasoning", "vendor-b", "model-b", 0.9, 0, 0),
    )

    decision = select_provider(profile, providers, policy)

    assert decision.provider is providers[1]
    assert decision.profile is profile
    assert decision.eligible_provider_ids == ("basic", "reasoning")
    assert decision.score == pytest.approx(0.9)


class ProviderSelectionTests(unittest.TestCase):
    def test_selects_expected_providers_for_characterization_cases(self) -> None:
        profiles = {
            "fast": QueryProfile(
                reasoning_depth=0.05,
                latency_sensitivity=0.95,
                cost_sensitivity=0.95,
            ),
            "balanced": QueryProfile(
                reasoning_depth=0.50,
                latency_sensitivity=0.50,
                cost_sensitivity=0.50,
            ),
            "reasoning": QueryProfile(
                reasoning_depth=0.95,
                latency_sensitivity=0.10,
                cost_sensitivity=0.10,
            ),
        }
        expected = {
            "fast": ("mock-fast", 0.47675),
            "balanced": ("mock-balanced", 0.36875),
            "reasoning": ("mock-reasoning", 0.48525),
        }

        for case, profile in profiles.items():
            with self.subTest(case=case):
                decision = select_provider(
                    profile,
                    DEFAULT_PROVIDERS,
                    DEFAULT_POLICY,
                )
                provider_id, score = expected[case]
                self.assertEqual(decision.provider.id, provider_id)
                self.assertAlmostEqual(decision.score, score)

    def test_uses_provider_id_as_a_deterministic_tie_breaker(self) -> None:
        profile = QueryProfile(
            reasoning_depth=0,
            latency_sensitivity=0,
            cost_sensitivity=0,
        )
        providers = (
            ProviderProfile("z", "provider", "model", 1, 1, 1),
            ProviderProfile("a", "provider", "model", 1, 1, 1),
        )

        decision = select_provider(profile, providers, DEFAULT_POLICY)

        self.assertEqual(decision.provider.id, "a")

    def test_rejects_an_empty_provider_registry(self) -> None:
        profile = QueryProfile(
            reasoning_depth=0.5,
            latency_sensitivity=0.5,
            cost_sensitivity=0.5,
        )

        with self.assertRaisesRegex(ValueError, "No providers configured"):
            select_provider(profile, (), DEFAULT_POLICY)


if __name__ == "__main__":
    unittest.main()
