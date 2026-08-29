from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_router.approval import (
    ApprovalEvaluationError,
    ApprovalEvaluator,
    ApprovalGate,
    ApprovalRejectedError,
    MockApprovalLLM,
    load_policy_json,
)
from llm_router.models import ApprovalPolicy, ApprovalProfile
from utils.scribe import Scribe


class StaticLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs: object) -> str:
        self.calls.append(prompt)
        return self.response


class StaticApprovalClassifier:
    def __init__(
        self,
        profile: ApprovalProfile | None = None,
        error: Exception | None = None,
    ) -> None:
        self.profile = profile
        self.error = error

    def classify(self, query: str) -> ApprovalProfile:
        if self.error is not None:
            raise self.error
        assert self.profile is not None
        return self.profile


def policy() -> ApprovalPolicy:
    return ApprovalPolicy.model_validate(
        {
            "version": 2,
            "algorithm_version": 1,
            "approval_rubric": "Approve safe requests.",
            "review_threshold": 0.35,
            "reject_threshold": 0.75,
            "uncertainty_weight": 0.2,
            "risk_weights": {
                "personal_info": 0.85,
                "medical_records": 0.9,
                "cyber_exploits": 1.0,
                "illegal_acts": 1.0,
                "harmful_materials": 1.0,
            },
        }
    )


def profile(*, harmful: float, uncertainty: float = 0.0) -> ApprovalProfile:
    return ApprovalProfile(
        personal_info_risk=0.0,
        medical_records_risk=0.0,
        cyber_exploits_risk=0.0,
        illegal_acts_risk=0.0,
        harmful_materials_risk=harmful,
        uncertainty=uncertainty,
    )


class ApprovalPolicyTests(unittest.TestCase):
    def test_loads_the_repository_policy(self) -> None:
        result = load_policy_json("approval-policy.json")

        self.assertEqual(result.version, 2)
        self.assertEqual(result.algorithm_version, 1)
        self.assertEqual(result.risk_weights.harmful_materials, 1.0)

    def test_rejects_v1_malformed_and_invalid_policies(self) -> None:
        temp_dir = Path(self.enterContext(TemporaryDirectory()))
        cases = {
            "malformed.json": "not json",
            "v1.json": json.dumps({"version": 1}),
            "thresholds.json": json.dumps(
                {
                    **policy().model_dump(mode="json"),
                    "review_threshold": 0.8,
                    "reject_threshold": 0.7,
                }
            ),
        }
        for filename, contents in cases.items():
            with self.subTest(filename=filename):
                path = temp_dir / filename
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_policy_json(path)

        with self.assertRaises(ValueError):
            load_policy_json(temp_dir / "missing.json")


class ApprovalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = Path(self.enterContext(TemporaryDirectory()))
        self.scribe = Scribe(temp_dir / "audit.db")
        self.policy = policy()

    def gate(
        self,
        classifier: StaticApprovalClassifier,
        reviewer: StaticLLM | MockApprovalLLM,
    ) -> ApprovalGate:
        return ApprovalGate(
            policy=self.policy,
            classifier=classifier,
            evaluator=ApprovalEvaluator(reviewer),
            scribe=self.scribe,
        )

    def test_auto_approves_low_risk_without_review(self) -> None:
        reviewer = StaticLLM('{"status":"rejected","reason":"must not run"}')

        result = self.gate(
            StaticApprovalClassifier(profile(harmful=0.2)), reviewer
        ).evaluate("query", "request-1")

        self.assertEqual(result.decision.status, "approved")
        self.assertEqual(result.decision.action, "auto_approve")
        self.assertEqual(result.decision.decided_by, "policy")
        self.assertEqual(reviewer.calls, [])

    def test_reviewer_approves_a_review_band_request(self) -> None:
        reviewer = StaticLLM(
            '{"status":"approved","reason":"Allowed by the rubric."}'
        )

        result = self.gate(
            StaticApprovalClassifier(profile(harmful=0.5)), reviewer
        ).evaluate("ambiguous query", "request-1")

        self.assertEqual(result.decision.status, "approved")
        self.assertEqual(result.decision.action, "review")
        self.assertEqual(result.decision.decided_by, "reviewer")
        self.assertEqual(len(reviewer.calls), 1)
        self.assertIn('"harmful_materials_risk": 0.5', reviewer.calls[0])

    def test_score_based_mock_reviewer_rejects_review_band_request(self) -> None:
        reviewer = MockApprovalLLM(reject_at=0.55)

        with self.assertRaises(ApprovalRejectedError) as raised:
            self.gate(
                StaticApprovalClassifier(profile(harmful=0.6)), reviewer
            ).evaluate("query", "request-1")

        self.assertEqual(raised.exception.decision.action, "review")
        self.assertEqual(raised.exception.decision.decided_by, "reviewer")
        self.assertEqual(len(reviewer.calls), 1)

    def test_auto_rejects_high_risk_without_review(self) -> None:
        reviewer = StaticLLM('{"status":"approved","reason":"must not run"}')

        with self.assertRaises(ApprovalRejectedError) as raised:
            self.gate(
                StaticApprovalClassifier(profile(harmful=0.9)), reviewer
            ).evaluate("query", "request-1")

        decision = raised.exception.decision
        self.assertEqual(decision.action, "auto_reject")
        self.assertEqual(decision.dominant_risk, "harmful_materials")
        self.assertEqual(decision.decided_by, "policy")
        self.assertEqual(reviewer.calls, [])
        record = self.scribe.verify()[-1]
        self.assertEqual(record.payload["profile"]["harmful_materials_risk"], 0.9)
        self.assertEqual(record.payload["algorithm_version"], 1)

    def test_failed_classification_remains_pending(self) -> None:
        reviewer = StaticLLM('{"status":"approved","reason":"must not run"}')

        with self.assertRaises(ApprovalEvaluationError) as raised:
            self.gate(
                StaticApprovalClassifier(error=ValueError("invalid profile")),
                reviewer,
            ).evaluate("query", "request-1")

        self.assertEqual(raised.exception.decision.status, "pending")
        self.assertEqual(
            [record.event_type for record in self.scribe.verify()],
            ["approval_pending", "approval_evaluation_failed"],
        )


if __name__ == "__main__":
    unittest.main()
