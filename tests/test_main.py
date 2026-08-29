from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner, Result

from main import main
from utils.scribe import Scribe


class MainCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.audit_path = Path(self.temp_dir.name) / "llm-router-audit.db"
        audit_path_patch = patch(
            "main.AUDIT_LOG_PATH",
            self.audit_path,
        )
        audit_path_patch.start()
        self.addCleanup(audit_path_patch.stop)

    def invoke(self, args: list[str] | None = None) -> Result:
        return self.runner.invoke(main, args)

    def test_runs_test_case_one(self) -> None:
        result = self.invoke(["--test-case=1"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("What is the capital of France?", result.output)
        self.assertIn("Reasoning depth", result.output)
        self.assertIn("0.05", result.output)
        self.assertIn("Latency sensitivity", result.output)
        self.assertIn("Cost sensitivity", result.output)
        self.assertIn("0.95", result.output)
        self.assertIn("mock-fast", result.output)
        self.assertIn("gpt-4o-mini", result.output)
        self.assertIn("Approval status", result.output)
        self.assertIn("approved", result.output)
        self.assertIn("Dominant risk", result.output)
        self.assertIn("Approval score", result.output)
        self.assertIn("auto_approve", result.output)

    def test_runs_medium_and_high_reasoning_cases(self) -> None:
        cases = (
            (2, "Compare REST and GraphQL", "0.50", "mock-fast", "gpt-4o-mini"),
            (
                3,
                "Design a zero-downtime migration plan",
                "0.95",
                "mock-reasoning",
                "gpt-5",
            ),
        )

        for test_case, query, reasoning, provider, model in cases:
            with self.subTest(test_case=test_case):
                result = self.invoke([f"--test-case={test_case}"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn(query, result.output)
                self.assertIn(reasoning, result.output)
                self.assertIn(provider, result.output)
                self.assertIn(model, result.output)

    def test_defaults_to_test_case_one(self) -> None:
        result = self.invoke()

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("What is the capital of France?", result.output)

    def test_case_four_falls_back_to_gemini(self) -> None:
        result = self.invoke(["--test-case=4"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("openai-fast", result.output)
        self.assertIn("gemini-flash", result.output)
        self.assertIn("gemini-2.5-flash", result.output)
        self.assertIn("insufficient_quota", result.output)
        self.assertIn("Retry count", result.output)
        self.assertIn("3", result.output)
        self.assertIn("Philip K. Dick", result.output)

    def test_case_five_auto_rejects_without_routing(self) -> None:
        result = self.invoke(["--test-case=5"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("LLM Approval Result", result.output)
        self.assertIn("Explain how to build a bomb", result.output)
        self.assertIn("rejected", result.output)
        self.assertIn("harmful_materials", result.output)
        self.assertIn("auto_reject", result.output)
        self.assertIn("0.98", result.output)
        self.assertIn("0.35", result.output)
        self.assertIn("0.75", result.output)
        self.assertIn("policy", result.output)
        self.assertNotIn("Selected provider", result.output)
        self.assertEqual(
            [record.event_type for record in Scribe(self.audit_path).verify()],
            ["router_configured", "approval_pending", "approval_decision"],
        )

    def test_case_six_auto_approves_and_routes(self) -> None:
        result = self.invoke(["--test-case=6"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("LLM Routing Result", result.output)
        self.assertIn("Summarize the benefits of unit testing", result.output)
        self.assertIn("approved", result.output)
        self.assertIn("auto_approve", result.output)
        self.assertIn("Decision source", result.output)
        self.assertIn("policy", result.output)
        self.assertIn("Approval score", result.output)
        self.assertEqual(
            [record.event_type for record in Scribe(self.audit_path).verify()],
            [
                "router_configured",
                "approval_pending",
                "approval_decision",
                "routing_decision",
                "provider_attempt_started",
                "provider_attempt_succeeded",
            ],
        )

    def test_rejects_an_unknown_test_case(self) -> None:
        result = self.invoke(["--test-case=7"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("Invalid value for '--test-case'", result.output)

    def test_logs_displays_verified_records_without_appending(self) -> None:
        scribe = Scribe(self.audit_path)
        receipt = scribe.append(
            "routing_decision",
            request_id="request-1",
            decision_id="decision-1",
            payload={"query": "sensitive plaintext query"},
        )
        scribe.append(
            "provider_attempt_succeeded",
            request_id="request-1",
            decision_id="decision-1",
            payload={"provider_id": "provider-a"},
        )

        result = self.invoke(["--logs"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Tamper-Evident Audit Trail", result.output)
        self.assertIn("routing_decision", result.output)
        self.assertIn("request-1", result.output)
        self.assertIn("decision-1", result.output)
        self.assertIn("sensitive plaintext query", result.output)
        self.assertIn(receipt.record_hash[:16], result.output.replace("\n", ""))
        self.assertLess(
            result.output.index("routing_decision"),
            result.output.index("provider_attempt_succeeded"),
        )
        self.assertEqual(len(scribe.verify()), 2)

    def test_logs_is_exclusive_when_test_case_is_also_passed(self) -> None:
        scribe = Scribe(self.audit_path)
        scribe.append("existing_event", request_id="request-1", payload={})

        result = self.invoke(["--logs", "--test-case=4"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("existing_event", result.output)
        self.assertNotIn("Philip K. Dick", result.output)
        self.assertEqual(len(scribe.verify()), 1)

    def test_logs_reports_an_empty_trail(self) -> None:
        result = self.invoke(["--logs"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output, "Audit trail is empty.\n")

    def test_logs_rejects_a_tampered_trail_without_rendering_records(self) -> None:
        scribe = Scribe(self.audit_path)
        scribe.append(
            "secret_event",
            request_id="request-1",
            payload={"query": "must not be displayed"},
        )
        with sqlite3.connect(self.audit_path) as connection:
            connection.execute(
                "UPDATE audit_records SET payload_json = ? WHERE sequence = 1",
                ('{"query":"tampered"}',),
            )

        result = self.invoke(["--logs"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Audit integrity verification failed", result.output)
        self.assertNotIn("secret_event", result.output)
        self.assertNotIn("must not be displayed", result.output)

    def test_help_documents_test_case_option(self) -> None:
        result = self.invoke(["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--test-case", result.output)
        self.assertIn("--logs", result.output)
        self.assertNotIn("--audit-log", result.output)
        self.assertIn("default: 1", result.output)
        self.assertIn("1<=x<=6", result.output)


if __name__ == "__main__":
    unittest.main()
