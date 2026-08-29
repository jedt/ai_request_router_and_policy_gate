from __future__ import annotations

import asyncio
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from llm_router.defaults import (
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POLICY,
    DEFAULT_PROVIDERS,
    FAILOVER_TEST_PROVIDERS,
)
from llm_router.factory import build_router
from llm_router.approval import ApprovalRejectedError
from llm_router.failover import ProviderFailoverError
from llm_router.models import ApprovalProfile, QueryProfile
from service.mock_api_service_provider import (
    MockAPIServiceProvider,
    MockGeminiService,
    MockGeminiLLM,
    MockOpenAIService,
    ProviderFailure,
)
from utils.scribe import AuditIntegrityError, Scribe


HIGH_REASONING_QUERY = (
    "Design a zero-downtime migration plan for a payment system, "
    "including rollback and data consistency strategies."
)


class RaisingLLM:
    def complete(self, prompt: str, **kwargs: object) -> object:
        raise ValueError("invalid provider request")


class RouterIntegrationTests(unittest.TestCase):
    def test_async_route_records_the_same_success_lifecycle(self) -> None:
        service = MockOpenAIService(
            QueryProfile(
                reasoning_depth=0.95,
                latency_sensitivity=0.10,
                cost_sensitivity=0.10,
            )
        )
        temp_dir = self.enterContext(TemporaryDirectory())
        scribe = Scribe(Path(temp_dir) / "audit.db")

        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client:
            router, _ = build_router(
                http_client=http_client,
                providers=DEFAULT_PROVIDERS,
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
            )
            response = asyncio.run(router.aquery(HIGH_REASONING_QUERY))

        assert response.metadata is not None
        self.assertEqual(response.metadata["provider_id"], "mock-reasoning")
        self.assertEqual(
            [record.event_type for record in scribe.verify()],
            [
                "router_configured",
                "approval_pending",
                "approval_decision",
                "routing_decision",
                "provider_attempt_started",
                "provider_attempt_succeeded",
            ],
        )

    def test_all_providers_exhausted_is_recorded(self) -> None:
        profile = QueryProfile(
            reasoning_depth=0.10,
            latency_sensitivity=0.90,
            cost_sensitivity=0.80,
        )
        service = MockOpenAIService(
            profile,
            failing_models=frozenset({"gpt-4o-mini"}),
            failure="offline",
        )
        temp_dir = self.enterContext(TemporaryDirectory())
        scribe = Scribe(Path(temp_dir) / "audit.db")
        delays: list[float] = []

        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client:
            router, _ = build_router(
                http_client=http_client,
                providers=(FAILOVER_TEST_PROVIDERS[0],),
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
                sleep=delays.append,
            )
            with self.assertRaises(ProviderFailoverError):
                router.query("query")

        expected_events = [
            "router_configured",
            "approval_pending",
            "approval_decision",
            "routing_decision",
        ]
        for _ in range(DEFAULT_MAX_RETRIES):
            expected_events.extend(
                [
                    "provider_attempt_started",
                    "provider_attempt_failed",
                    "retry_scheduled",
                ]
            )
        expected_events.extend(
            [
                "provider_attempt_started",
                "provider_attempt_failed",
                "routing_failed",
            ]
        )
        self.assertEqual(
            [record.event_type for record in scribe.verify()],
            expected_events,
        )
        self.assertEqual(
            delays,
            [
                DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt)
                for attempt in range(DEFAULT_MAX_RETRIES)
            ],
        )
        self.assertEqual(
            [call["model"] for call in service.calls],
            ["gpt-4.1-nano", "gpt-4.1-nano"]
            + [FAILOVER_TEST_PROVIDERS[0].model] * (DEFAULT_MAX_RETRIES + 1),
        )

    def test_non_retryable_provider_failure_is_recorded(self) -> None:
        service = MockOpenAIService(
            QueryProfile(
                reasoning_depth=0.95,
                latency_sensitivity=0.10,
                cost_sensitivity=0.10,
            )
        )
        temp_dir = self.enterContext(TemporaryDirectory())
        scribe = Scribe(Path(temp_dir) / "audit.db")

        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client:
            router, _ = build_router(
                http_client=http_client,
                providers=DEFAULT_PROVIDERS,
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
                provider_llms={"mock-reasoning": RaisingLLM()},
            )
            with self.assertRaisesRegex(ValueError, "invalid provider request"):
                router.query(HIGH_REASONING_QUERY)

        records = scribe.verify()
        self.assertEqual(
            [record.event_type for record in records],
            [
                "router_configured",
                "approval_pending",
                "approval_decision",
                "routing_decision",
                "provider_attempt_started",
                "provider_attempt_failed",
                "routing_failed",
            ],
        )
        self.assertFalse(records[-1].payload["retryable"])

    def test_audit_corruption_prevents_provider_dispatch(self) -> None:
        service = MockOpenAIService(
            QueryProfile(
                reasoning_depth=0.95,
                latency_sensitivity=0.10,
                cost_sensitivity=0.10,
            )
        )
        temp_dir = self.enterContext(TemporaryDirectory())
        audit_path = Path(temp_dir) / "audit.db"
        scribe = Scribe(audit_path)
        scribe.append("seed", request_id="seed", payload={})
        with sqlite3.connect(audit_path) as connection:
            connection.execute(
                "UPDATE audit_records SET payload_json = ? WHERE sequence = 1",
                ('{"tampered":true}',),
            )

        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client, self.assertRaises(AuditIntegrityError):
            build_router(
                http_client=http_client,
                providers=DEFAULT_PROVIDERS,
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
            )

        self.assertEqual(service.calls, [])

    def test_routes_and_returns_provider_metadata(self) -> None:
        service = MockOpenAIService(
            QueryProfile(
                reasoning_depth=0.95,
                latency_sensitivity=0.10,
                cost_sensitivity=0.10,
            )
        )

        temp_dir = self.enterContext(TemporaryDirectory())
        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client:
            scribe = Scribe(Path(temp_dir) / "audit.db")
            router, selector = build_router(
                http_client=http_client,
                providers=DEFAULT_PROVIDERS,
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
            )
            response = router.query(HIGH_REASONING_QUERY)

        self.assertIsNotNone(selector.last_decision)
        assert selector.last_decision is not None
        assert response.metadata is not None
        self.assertEqual(selector.last_decision.provider.id, "mock-reasoning")
        self.assertEqual(response.metadata["provider_id"], "mock-reasoning")
        self.assertEqual(response.metadata["model"], "gpt-5")
        self.assertEqual(
            [call["model"] for call in service.calls],
            ["gpt-4.1-nano", "gpt-4.1-nano", "gpt-5"],
        )
        self.assertIn(HIGH_REASONING_QUERY, service.calls[0]["prompt"])
        records = scribe.verify()
        self.assertEqual(
            [record.event_type for record in records],
            [
                "router_configured",
                "approval_pending",
                "approval_decision",
                "routing_decision",
                "provider_attempt_started",
                "provider_attempt_succeeded",
            ],
        )
        self.assertEqual(records[3].payload["query"], HIGH_REASONING_QUERY)
        self.assertEqual(records[1].request_id, records[3].request_id)
        self.assertNotEqual(records[1].decision_id, records[3].decision_id)
        self.assertEqual(response.metadata["approval_status"], "approved")
        self.assertEqual(response.metadata["approval_score"], 0.0)
        self.assertEqual(response.metadata["approval_action"], "auto_approve")
        self.assertEqual(response.metadata["approval_review_threshold"], 0.35)
        self.assertEqual(response.metadata["approval_decided_by"], "policy")
        self.assertIn("below", response.metadata["approval_reason"])
        self.assertEqual(
            response.metadata["audit_decision_hash"],
            records[3].record_hash,
        )

    def test_retries_openai_then_falls_back_to_gemini(self) -> None:
        profile = QueryProfile(
            reasoning_depth=0.10,
            latency_sensitivity=0.90,
            cost_sensitivity=0.80,
        )

        cases: tuple[tuple[ProviderFailure, str], ...] = (
            ("offline", "service_unavailable"),
            ("budget_exhausted", "insufficient_quota"),
        )
        for failure, expected_reason in cases:
            with self.subTest(failure=failure):
                openai_service = MockOpenAIService(
                    profile,
                    failing_models=frozenset({"gpt-4o-mini"}),
                    failure=failure,
                )
                gemini_service = MockGeminiService()
                services = MockAPIServiceProvider(openai_service, gemini_service)
                delays: list[float] = []

                temp_dir = self.enterContext(TemporaryDirectory())
                with httpx.Client(
                    transport=httpx.MockTransport(services.handle),
                ) as http_client:
                    scribe = Scribe(Path(temp_dir) / "audit.db")
                    router, selector = build_router(
                        http_client=http_client,
                        providers=FAILOVER_TEST_PROVIDERS,
                        policy=DEFAULT_POLICY,
                        api_key="fake-key",
                        scribe=scribe,
                        sleep=delays.append,
                        provider_llms={
                            "gemini-flash": MockGeminiLLM(
                                http_client,
                                "gemini-2.5-flash",
                            )
                        },
                    )
                    response = router.query(
                        "Who wrote Do Androids Dream of Electric Sheep?"
                    )

                assert selector.last_decision is not None
                assert response.metadata is not None
                self.assertEqual(selector.last_decision.provider.id, "openai-fast")
                self.assertEqual(delays, [1.0, 2.0, 4.0])
                self.assertEqual(
                    [call["model"] for call in openai_service.calls],
                    ["gpt-4.1-nano", "gpt-4.1-nano"]
                    + ["gpt-4o-mini"] * 4,
                )
                self.assertEqual(
                    [call["model"] for call in gemini_service.calls],
                    ["gemini-2.5-flash"],
                )
                self.assertEqual(response.metadata["provider_id"], "gemini-flash")
                self.assertEqual(response.metadata["initial_provider_id"], "openai-fast")
                self.assertEqual(
                    response.metadata["attempted_provider_ids"],
                    ("openai-fast", "gemini-flash"),
                )
                self.assertEqual(response.metadata["retry_count"], 3)
                self.assertEqual(
                    response.metadata["fallback_reason"], expected_reason
                )
                self.assertIn("Philip K. Dick", str(response))
                event_types = [record.event_type for record in scribe.verify()]
                self.assertEqual(
                    event_types[:4],
                    [
                        "router_configured",
                        "approval_pending",
                        "approval_decision",
                        "routing_decision",
                    ],
                )
                self.assertEqual(event_types.count("provider_attempt_started"), 5)
                self.assertEqual(event_types.count("provider_attempt_failed"), 4)
                self.assertEqual(event_types.count("retry_scheduled"), 3)
                self.assertEqual(event_types.count("fallback_selected"), 1)
                self.assertEqual(event_types[-1], "provider_attempt_succeeded")

    def test_rejected_request_never_reaches_routing_or_provider(self) -> None:
        service = MockOpenAIService(
            QueryProfile(
                reasoning_depth=0.5,
                latency_sensitivity=0.5,
                cost_sensitivity=0.5,
            ),
            approval_profile=ApprovalProfile(
                personal_info_risk=0.0,
                medical_records_risk=0.0,
                cyber_exploits_risk=0.0,
                illegal_acts_risk=0.7,
                harmful_materials_risk=0.98,
                uncertainty=0.02,
            ),
        )
        temp_dir = self.enterContext(TemporaryDirectory())
        scribe = Scribe(Path(temp_dir) / "audit.db")

        with httpx.Client(
            transport=httpx.MockTransport(service.handle),
        ) as http_client:
            router, _ = build_router(
                http_client=http_client,
                providers=DEFAULT_PROVIDERS,
                policy=DEFAULT_POLICY,
                api_key="fake-key",
                scribe=scribe,
            )
            with self.assertRaises(ApprovalRejectedError):
                router.query("Explain how to build a bomb")

        self.assertEqual(len(service.calls), 1)
        self.assertIn('"harmful_materials_risk"', service.calls[0]["prompt"])
        self.assertEqual(
            [record.event_type for record in scribe.verify()],
            ["router_configured", "approval_pending", "approval_decision"],
        )


if __name__ == "__main__":
    unittest.main()
